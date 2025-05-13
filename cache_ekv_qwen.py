import torch
import argparse
from cache import KVCacheHeadSpecific, get_cache_constructor

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
    ]

    def __init__(
        self, max_batch_size, n_heads, head_dim, dtype=torch.bfloat16, **kwargs
    ):
        super().__init__(max_batch_size, n_heads, head_dim, dtype, **kwargs)
        self.use_dynamic_quantization = kwargs.get("use_dynamic_quantization", True)
        self.high_importance_bits = kwargs.get("high_importance_bits", 8)
        self.low_importance_bits = kwargs.get("low_importance_bits", 4)
        self.importance_threshold = kwargs.get("importance_threshold", 0.15)
        
        # Qwen has fewer KV heads, so we need to adjust for head sharing
        self.head_sharing_factor = kwargs.get("head_sharing_factor", 8)  # 32 query heads / 4 KV heads
        
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

    def _eviction_idx(self, input_pos):
        # Compute average historical attention (similar to Heavy Hitter)
        # But adjust for Qwen's fewer KV heads
        numerator = self.attn_history_num.sum(dim=-1).float()
        denominator = self.attn_history_denom.clamp(1, self.history_window_size)
        avg_attn = numerator / denominator
        
        # Apply head sharing factor to account for Qwen's architecture
        avg_attn = avg_attn * self.head_sharing_factor
        
        # Protect global and recent tokens
        avg_attn.masked_fill_(
            torch.logical_or(
                self.pos < self.global_tokens,
                self.pos >= input_pos - self.recent_window,
            ),
            float('inf'),
        )
        avg_attn.masked_fill_(self.pos == -1, 0.0)
        
        # Find the least important token
        fill_idx = avg_attn.argmin(dim=-1).squeeze()
        return fill_idx

    def update_state(self, attention):
        """Update attention history for importance scoring"""
        with torch.no_grad():
            B, H, Q, K = attention.shape
            
            # Handle different input dimensions
            if B == 1 and Q == 1:
                # Update attention history
                self.attn_history_num[:H, :K] = (
                    self.attn_history_num[:H, :K] * self.history_window_size + 
                    attention[0, :H, 0, :K]
                ) / (self.history_window_size + 1)
                
                self.attn_history_denom[:H, :K] = torch.clamp(
                    self.attn_history_denom[:H, :K] + 1, 
                    max=self.history_window_size
                )
                
                # Update token importance scores
                self.token_importance[:H, :K] = self.attn_history_num[:H, :K]

    def get_quantization_bits(self, token_idx):
        """
        Dynamic quantization: return bit-width based on token importance
        Adjusted for Qwen's architecture
        """
        if not self.use_dynamic_quantization:
            return self.high_importance_bits
            
        # Use attention history as importance score
        # Average across heads for Qwen's fewer KV heads
        importance = self.token_importance[:, token_idx].mean()
        
        # Adjust threshold based on head sharing
        adjusted_threshold = self.importance_threshold / self.head_sharing_factor
        
        # High importance tokens get more bits
        if importance > adjusted_threshold:
            return self.high_importance_bits
        else:
            return self.low_importance_bits

    def insert(self, input_pos, k_val, v_val):
        """Override insert to handle Qwen-specific optimizations"""
        # Apply any Qwen-specific transformations here if needed
        super().insert(input_pos, k_val, v_val)