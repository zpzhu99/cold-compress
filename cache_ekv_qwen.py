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

class KVCacheEKVQwen(KVCacheHeadSpecific):
    """
    EKV-inspired cache optimized for Qwen models that combines 
    attention-based eviction with dynamic quantization based on token importance
    and QuaRot Hadamard transformations.
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
            (self.n_heads, self.max_cache_length), device="cuda", dtype=dtype
        )
        self.attn_history_denom = torch.zeros(
            (self.n_heads, self.max_cache_length), device="cuda", dtype=dtype
        )
        self.history_window_size = kwargs.get("history_window_size", 1024)
        
        # Token importance scores for dynamic quantization
        self.token_importance = torch.zeros(
            (self.n_heads, self.max_cache_length), device="cuda", dtype=dtype
        )
        
        # Store the cache bits for potential use
        self.cache_bits = kwargs.get("cache_bits", None)
        
        # QuaRot specific initialization
        self.use_quarot = kwargs.get("use_quarot", False)
        if self.use_quarot:
            # Store the head dimension for Hadamard matrix
            self.hadamard_dim = head_dim
            
            # Create Hadamard matrices with the same dtype as the cache
            self.H = create_hadamard_matrix(head_dim, device="cuda", dtype=dtype)
            
            # For randomization (as in QuaRot paper)
            diagonal_values = torch.randint(0, 2, (head_dim,), device="cuda", dtype=torch.int32) * 2 - 1
            self.D = torch.diag(diagonal_values.to(dtype))
            self.HD = torch.matmul(self.H, self.D)
            self.HD_inv = self.HD.t()  # Inverse is transpose for orthogonal matrix
            
            self.clipping_ratio = kwargs.get("clipping_ratio", 0.9)

    def apply_hadamard_transform(self, x):
        """Apply Hadamard transformation to reduce outliers"""
        if x.numel() == 0:  # Handle empty tensors
            return x
            
        original_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        
        # Apply HD transformation  
        x_transformed = torch.matmul(x_flat, self.HD)
        
        return x_transformed.reshape(original_shape)
    
    def apply_inverse_hadamard(self, x):
        """Apply inverse Hadamard transformation"""
        if x.numel() == 0:  # Handle empty tensors
            return x
            
        original_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        
        # Apply inverse HD transformation
        x_original = torch.matmul(x_flat, self.HD_inv)
        
        return x_original.reshape(original_shape)

    def quantize_cache(self):
        """Override to apply Hadamard before quantization"""
        if self.use_quarot and self.quantize:
            # Apply Hadamard transformation before quantization
            k_transformed = self.apply_hadamard_transform(self.k_cache)
            v_transformed = self.apply_hadamard_transform(self.v_cache)
            
            # Temporarily store transformed values for quantization
            original_k = self.k_cache.clone()
            original_v = self.v_cache.clone()
            
            self.k_cache = k_transformed
            self.v_cache = v_transformed
            
            # Call parent's quantization
            super().quantize_cache()
            
            # Store the scales but keep original values for retrieval
            self.k_cache_quantized = self.k_cache.clone()
            self.v_cache_quantized = self.v_cache.clone()
            
            # Restore original values
            self.k_cache = original_k
            self.v_cache = original_v
        else:
            super().quantize_cache()

    def dequantize_cache(self):
        """Override to apply inverse Hadamard after dequantization"""
        if self.use_quarot and self.quantize:
            # Get quantized values
            if hasattr(self, 'k_cache_quantized'):
                self.k_cache = self.k_cache_quantized
                self.v_cache = self.v_cache_quantized
            
            # Dequantize
            super().dequantize_cache()
            
            # Apply inverse Hadamard to get back to original space
            self.k_cache = self.apply_inverse_hadamard(self.k_cache)
            self.v_cache = self.apply_inverse_hadamard(self.v_cache)
        else:
            super().dequantize_cache()

    def _fill(self, input_pos, k_val, v_val, fill_idxs, **kwargs):
        """Store original values, transform only for quantization"""
        # Always store original values
        super()._fill(input_pos, k_val, v_val, fill_idxs, **kwargs)

    def _eviction_idx(self, input_pos):
        """Determine which token to evict per head"""
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

    def return_kv_cache(self):
        """Return the KV cache values"""
        # For QuaRot, we store original values and transform only during quantization
        # So we can return the cache as-is
        return super().return_kv_cache()

    @property
    def size(self):
        """Return the effective size of the cache"""
        return (self.pos >= 0).sum(dim=1).max().item()