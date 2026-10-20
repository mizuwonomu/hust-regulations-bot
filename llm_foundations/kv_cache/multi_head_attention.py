"""Tính Multi head Attention với KV Cache của từng head."""
import torch
from torch import nn

from llm_foundations.cached_self_attention import CachedCausalSelfAttention


class CachedMultiHeadAttention(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int,
        d_k: int,
        d_v: int
    ) -> None:
        """Khai báo từng Attention head, và trọng số projection.
        
        Args:
            d_model: Kích thước đặc trưng đầu vào.
            num_heads: Số Attention head trong 1 block Transformer.
            d_k: Kích thước đặc trưng của ma trận Key.
            d_v: Kích thước đặc trưng của ma trận Value.    
        """
        super().__init__()
        self.cached_heads = nn.ModuleList([CachedCausalSelfAttention(d_model, d_k, d_v) for _ in range(num_heads)]) # Tạo các attention heads có sử dụng KV Cache
        self.w_o = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_heads*d_v, d_model)))

    def prefill(
        self,
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Tính toán trước các token của prompt input, tính ra output attention của từng head,
        attention weights và lưu bộ KV Cache ở từng head. Sau đó, concat output và stack weights,
        lưu mọi KV Cache của các heads thành 1 list. Cuối cùng, tính toán với projection weights 
        để ra output cuối.
        
        Args:
            x: Input dùng để tính toán.
        """
        outputs = []
        weights = []
        kv_caches: list[tuple] = []

        for cache_head in self.cached_heads:
            output_h, weight_h, (K_old, V_old) = cache_head.prefill(x)

            outputs.append(output_h)
            weights.append(weight_h)

            kv_caches.append((K_old, V_old))

        concat_outputs = torch.concat(outputs, dim=1)
        full_head_weights = torch.stack(weights, dim=0)

        output = concat_outputs@self.w_o

        prefill_heads = output, full_head_weights, kv_caches
        return prefill_heads

    def decode(
        self,
        x_new: torch.Tensor,
        caches: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Decode phase, tính toán bộ Q,K,V của new query, nối KV vào bộ KV cache cũ của từng head,
        rồi tính attention trên từng head với bộ KV Cache đã nối. Sau đó, concat output và stack weights,
        lưu mọi KV Cache của các heads thành 1 list. Cuối cùng, tính toán với projection weights 
        để ra output cuối.

        Args:
            x_new: Hidden state mới mang shape [1, D].
            caches: List bộ KV Cache đã lưu ở các head trước.
        """
        new_outputs = []
        new_weights = []
        updated_caches: list[tuple] = []

        for cache_head, cache in zip(self.cached_heads, caches): # Duyệt vừa tính attention trên new query, vừa append new cache trên từng head
            output, attention_weights, (K_updated, V_updated) = cache_head.decode(x_new, cache)

            new_outputs.append(output)
            new_weights.append(attention_weights)

            updated_caches.append((K_updated, V_updated))

        concat_new_outputs = torch.concat(new_outputs, dim=1)
        full_new_head_weights = torch.stack(new_weights, dim=0)

        new_output = concat_new_outputs@self.w_o

        decode_heads = new_output, full_new_head_weights, updated_caches

        return decode_heads
        