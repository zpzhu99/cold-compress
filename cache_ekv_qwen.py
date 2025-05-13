# cache_ekv_qwen.py
import torch
import argparse
from cache import KVCacheHeadSpecific

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
        "cache_bits",  # Add this to inherited kwargs
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

    def _eviction_idx(self, input_pos):
        """Determine which token to evict"""
        # Check parent class implementation - it expects a single index
        # For head-specific caches, we need to return per-head indices
        
        # Calculate average attention scores across positions
        avg_scores = self.attn_history_num / self.attn_history_denom.clamp(min=1)
        
        # Apply head sharing factor
        avg_scores = avg_scores * self.head_sharing_factor
        
        # Mask to protect certain positions
        # Note: self.pos has shape [n_heads, max_cache_length]
        mask = torch.zeros_like(avg_scores)
        
        # Protect global tokens
        mask[self.pos < self.global_tokens] = float('inf')
        
        # Protect recent tokens
        if torch.is_tensor(input_pos):
            current_pos = input_pos.item() if input_pos.numel() == 1 else input_pos.max().item()
        else:
            current_pos = input_pos
        mask[self.pos >= current_pos - self.recent_window] = float('inf')
        
        # Invalid positions
        mask[self.pos == -1] = -float('inf')
        
        # Apply mask
        scores_with_mask = avg_scores + mask
        
        # Get minimum per head
        eviction_indices = scores_with_mask.argmin(dim=1)
        
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