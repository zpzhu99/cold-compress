# cache_ekv_qwen.py
import torch
import math
import argparse
from cache import KVCacheHeadSpecific

def create_hadamard_matrix(n, device='cuda', dtype=torch.float32):
    """Create a Hadamard matrix of size n x n"""
    if n == 1:
        return torch.tensor([[1.0]], device=device, dtype=dtype)
    
    # Ensure n is a power of 2
    n_pow2 = 2 ** math.ceil(math.log2(n))
    
    # Recursive construction
    H_half = create_hadamard_matrix(n_pow2 // 2, device, dtype)
    H = torch.zeros(n_pow2, n_pow2, device=device, dtype=dtype)
    H[:n_pow2//2, :n_pow2//2] = H_half
    H[:n_pow2//2, n_pow2//2:] = H_half
    H[n_pow2//2:, :n_pow2//2] = H_half
    H[n_pow2//2:, n_pow2//2:] = -H_half
    
    # Normalize and return only the needed size
    H = H[:n, :n] / math.sqrt(n)
    return H

def create_randomized_hadamard(n, device='cuda'):
    """Create a randomized Hadamard matrix as in QuaRot"""
    H = create_hadamard_matrix(n, device)
    
    # Random diagonal matrix for randomization
    D = torch.diag((torch.randint(0, 2, (n,), device=device).float() * 2 - 1))
    
    # Randomized Hadamard: D * H * D
    return torch.matmul(torch.matmul(D, H), D)

class KVCacheEKVQwen(KVCacheHeadSpecific):
    """
    EKV-inspired cache optimized for Qwen models that combines 
    attention-based eviction with dynamic quantization based on token importance.
    
    Note: Qwen-2 has 4 KV heads vs Llama-3's 8, which affects Heavy Hitter performance.
    """
    relevant_kwargs = [
        "max_cache_length",
        "max_seq_length",
        "global_tokens",
        "recent_window",
        "use_dynamic_quantization",
        "high_importance_bits",
        "low_importance_bits",
        "importance_threshold",
        "head_sharing_factor",
        "history_window_size",
        "cache_bits",
        "use_quarot",
        "clipping_ratio",
    ]

    def __init__(
        self, max_batch_size, n_heads, head_dim, dtype=torch.bfloat16, **kwargs
    ):
        # Make sure we pass all required kwargs to parent
        kwargs_for_parent = {}
        for key in ["max_cache_length", "max_seq_length", "global_tokens", "recent_window", "cache_bits"]:
            if key in kwargs:
                kwargs_for_parent[key] = kwargs[key]
        
        # Initialize parent class first
        super().__init__(max_batch_size, n_heads, head_dim, dtype, **kwargs_for_parent)
        
        # EKV specific attributes
        self.use_dynamic_quantization = kwargs.get("use_dynamic_quantization", True)
        self.high_importance_bits = kwargs.get("high_importance_bits", 8)
        self.low_importance_bits = kwargs.get("low_importance_bits", 4)
        self.importance_threshold = kwargs.get("importance_threshold", 0.15)
        
        # Qwen has fewer KV heads, so we need to adjust for head sharing
        self.head_sharing_factor = kwargs.get("head_sharing_factor", 8)
        
        # Initialize attention history for heavy hitter tracking
        self.attn_history_num = torch.zeros(
            (self.n_heads, self.max_cache_length), device="cuda"
        )
        self.attn_history_denom = torch.zeros(
            (self.n_heads, self.max_cache_length), device="cuda"
        )
        self.history_window_size = kwargs.get("history_window_size", 1024)
        
        # Token importance scores for dynamic quantization
        self.token_importance = torch.zeros(
            (self.n_heads, self.max_cache_length), device="cuda"
        )
        
        # Store the cache bits for potential use
        self.cache_bits = kwargs.get("cache_bits", None)
        
        # QuaRot specific initialization
        self.use_quarot = kwargs.get("use_quarot", False)
        if self.use_quarot:
            # Create Hadamard matrices with the same dtype as the cache
            self.H = create_hadamard_matrix(head_dim, device="cuda").to(dtype)
            # For randomization (as in QuaRot paper)
            self.D = torch.diag(torch.randint(0, 2, (head_dim,), device="cuda") * 2 - 1).float().to(dtype)
            self.HD = torch.matmul(self.H, self.D)
            self.HD_inv = self.HD.t()  # Inverse is transpose for orthogonal matrix
            
            self.clipping_ratio = kwargs.get("clipping_ratio", 0.9)
            
            # Storage for quantization scales with the same dtype
            self.register_buffer("k_quarot_scales", torch.ones((max_batch_size, n_heads, self.max_cache_length), dtype=dtype))
            self.register_buffer("v_quarot_scales", torch.ones((max_batch_size, n_heads, self.max_cache_length), dtype=dtype))

    def apply_hadamard_transform(self, x):
        """Apply Hadamard transformation to reduce outliers"""
        # x shape: [batch, heads, seq_len, head_dim]
        batch, heads, seq_len, dim = x.shape
        
        # Pad if necessary
        if dim < self.hadamard_dim:
            padding = self.hadamard_dim - dim
            x_padded = torch.nn.functional.pad(x, (0, padding))
        else:
            x_padded = x
        
        # Reshape for matrix multiplication
        x_flat = x_padded.reshape(-1, self.hadamard_dim)
        
        # Apply transformation
        x_transformed = torch.matmul(x_flat, self.H)
        
        # Reshape back and remove padding
        x_transformed = x_transformed.reshape(batch, heads, seq_len, self.hadamard_dim)
        if dim < self.hadamard_dim:
            x_transformed = x_transformed[:, :, :, :dim]
        
        return x_transformed
    
    def apply_inverse_hadamard(self, x):
        """Apply inverse Hadamard transformation during retrieval"""
        batch, heads, seq_len, dim = x.shape
        
        # Pad if necessary
        if dim < self.hadamard_dim:
            padding = self.hadamard_dim - dim
            x_padded = torch.nn.functional.pad(x, (0, padding))
        else:
            x_padded = x
        
        # Reshape for matrix multiplication
        x_flat = x_padded.reshape(-1, self.hadamard_dim)
        
        # Apply inverse transformation
        x_original = torch.matmul(x_flat, self.H_inv)
        
        # Reshape back and remove padding
        x_original = x_original.reshape(batch, heads, seq_len, self.hadamard_dim)
        if dim < self.hadamard_dim:
            x_original = x_original[:, :, :, :dim]
        
        return x_original
    
    def quantize_with_quarot(self, tensor, n_bits=4):
        """Quantize tensor using QuaRot approach with per-token quantization"""
        # tensor shape: [batch, heads, seq_len, head_dim]
        
        # Compute scale per token (across head_dim)
        max_vals = torch.quantile(tensor.abs().view(-1, tensor.shape[-1]), 
                                self.clipping_ratio, dim=-1, keepdim=True)
        max_vals = max_vals.view(tensor.shape[:-1] + (1,))
        
        # Compute scale
        scale = max_vals / (2**(n_bits-1) - 1)
        scale = scale.squeeze(-1)  # Remove last dimension for storage
        
        # Quantize
        quantized = torch.clamp(tensor / max_vals, -1.0, 1.0)
        quantized = (quantized * (2**(n_bits-1) - 1)).round()
        
        # Store as appropriate integer type
        if n_bits <= 8:
            quantized = quantized.to(torch.int8)
        else:
            quantized = quantized.to(torch.int16)
        
        return quantized, scale
    
    def dequantize_with_quarot(self, quantized, scale, n_bits=4):
        """Dequantize tensor"""
        # Reshape scale to match quantized dimensions
        scale = scale.unsqueeze(-1)
        
        # Dequantize
        dequantized = quantized.float() / (2**(n_bits-1) - 1)
        dequantized = dequantized * scale
        
        return dequantized
    
    def _fill(self, input_pos, k_val, v_val, fill_idxs, **kwargs):
        """Override to apply Hadamard transformation before storing"""
        
        if self.use_quarot:
            # Apply Hadamard transformation to reduce outliers
            k_transformed = self.apply_hadamard_transform(k_val)
            v_transformed = self.apply_hadamard_transform(v_val)
            
            # Determine quantization bits based on importance
            batch_idx = 0
            bits_per_position = []
            
            for idx in range(k_val.shape[2]):  # seq_len
                bits = self.get_quantization_bits(idx)
                bits_per_position.append(bits)
            
            # Quantize each position with appropriate bits
            k_quantized_list = []
            v_quantized_list = []
            k_scales_list = []
            v_scales_list = []
            
            for idx, bits in enumerate(bits_per_position):
                k_slice = k_transformed[:, :, idx:idx+1, :]
                v_slice = v_transformed[:, :, idx:idx+1, :]
                
                k_q, k_s = self.quantize_with_quarot(k_slice, bits)
                v_q, v_s = self.quantize_with_quarot(v_slice, bits)
                
                k_quantized_list.append(k_q)
                v_quantized_list.append(v_q)
                k_scales_list.append(k_s)
                v_scales_list.append(v_s)
            
            # Concatenate results
            k_quantized = torch.cat(k_quantized_list, dim=2)
            v_quantized = torch.cat(v_quantized_list, dim=2)
            
            # Store scales
            if isinstance(fill_idxs, torch.Tensor):
                for i, idx in enumerate(fill_idxs):
                    self.k_scales[:, :, idx] = k_scales_list[i].squeeze(2)
                    self.v_scales[:, :, idx] = v_scales_list[i].squeeze(2)
            else:
                self.k_scales[:, :, fill_idxs] = torch.stack(k_scales_list, dim=2).squeeze(3)
                self.v_scales[:, :, fill_idxs] = torch.stack(v_scales_list, dim=2).squeeze(3)
            
            # Store quantized values
            super()._fill(input_pos, k_quantized, v_quantized, fill_idxs, **kwargs)
        else:
            # No QuaRot - use standard fill
            super()._fill(input_pos, k_val, v_val, fill_idxs, **kwargs)
    
    def return_kv_cache(self):
        """Override to apply inverse Hadamard when retrieving from cache"""
        k_cache, v_cache, mask = super().return_kv_cache()
        
        if self.use_quarot and self.k_cache.dtype in [torch.int8, torch.int16]:
            # Dequantize each position with its stored scale and bits
            k_dequantized = []
            v_dequantized = []
            
            for idx in range(self.max_cache_length):
                # Get bits used for this position (simplified - could store this)
                bits = self.get_quantization_bits(idx)
                
                # Dequantize
                k_slice = k_cache[:, :, idx:idx+1, :]
                v_slice = v_cache[:, :, idx:idx+1, :]
                k_scale = self.k_scales[:, :, idx:idx+1]
                v_scale = self.v_scales[:, :, idx:idx+1]
                
                k_deq = self.dequantize_with_quarot(k_slice, k_scale, bits)
                v_deq = self.dequantize_with_quarot(v_slice, v_scale, bits)
                
                k_dequantized.append(k_deq)
                v_dequantized.append(v_deq)
            
            k_dequantized = torch.cat(k_dequantized, dim=2)
            v_dequantized = torch.cat(v_dequantized, dim=2)
            
            # Apply inverse Hadamard to restore original space
            k_original = self.apply_inverse_hadamard(k_dequantized)
            v_original = self.apply_inverse_hadamard(v_dequantized)
            
            return k_original, v_original, mask
        
        return k_cache, v_cache, mask

    def _eviction_idx(self, input_pos):
        """Determine which token to evict per head"""
        # For KVCacheHeadSpecific, we need to return indices per head
        
        # Calculate importance scores per head
        importance_scores = self.attn_history_num / self.attn_history_denom.clamp(min=1)
        
        # Apply head sharing factor
        importance_scores = importance_scores * self.head_sharing_factor
        
        # Create a mask to protect certain positions
        mask = torch.zeros_like(importance_scores)
        
        # Get current position
        if torch.is_tensor(input_pos):
            current_pos = input_pos.item() if input_pos.numel() == 1 else input_pos.max().item()
        else:
            current_pos = input_pos
        
        # self.pos has shape [batch_size, n_heads, max_cache_length]
        # We need to work with the first batch (index 0)
        batch_idx = 0
        
        # For each head, protect global and recent tokens
        for h in range(self.n_heads):
            # Protect global tokens
            mask[h, self.pos[batch_idx, h] < self.global_tokens] = float('inf')
            # Protect recent tokens  
            mask[h, self.pos[batch_idx, h] >= current_pos - self.recent_window] = float('inf')
            # Mark invalid positions
            mask[h, self.pos[batch_idx, h] == -1] = -float('inf')
        
        # Apply mask to scores
        masked_scores = importance_scores + mask
        
        # Find minimum score position for each head
        eviction_indices = masked_scores.argmin(dim=1)
        
        return eviction_indices

    def update_state(self, input_pos, k, v, is_prefill, attn, **kwargs):
        """Update attention history for importance scoring"""
        # Use only the attention tensor (attn) from the parameters
        with torch.no_grad():
            if attn is not None:
                B, H, Q, K = attn.shape
                
                # Handle different input dimensions
                if B == 1 and Q == 1:
                    # Update attention history
                    valid_k = min(K, self.max_cache_length)
                    valid_h = min(H, self.n_heads)
                    
                    self.attn_history_num[:valid_h, :valid_k] = (
                        self.attn_history_num[:valid_h, :valid_k] * self.history_window_size + 
                        attn[0, :valid_h, 0, :valid_k]
                    ) / (self.history_window_size + 1)
                    
                    self.attn_history_denom[:valid_h, :valid_k] = torch.clamp(
                        self.attn_history_denom[:valid_h, :valid_k] + 1, 
                        max=self.history_window_size
                    )
                    
                    # Update token importance scores
                    self.token_importance[:valid_h, :valid_k] = self.attn_history_num[:valid_h, :valid_k]

    def get_quantization_bits(self, token_idx):
        """Dynamic quantization based on token importance"""
        if not self.use_dynamic_quantization or self.cache_bits is not None:
            # If cache_bits is set globally, use that
            return self.cache_bits if self.cache_bits is not None else self.high_importance_bits
            
        importance = self.token_importance[:, token_idx].mean()
        adjusted_threshold = self.importance_threshold / self.head_sharing_factor
        
        if importance > adjusted_threshold:
            return self.high_importance_bits
        else:
            return self.low_importance_bits

    @property
    def size(self):
        """Return the effective size of the cache"""
        return (self.pos >= 0).sum(dim=1).max().item()