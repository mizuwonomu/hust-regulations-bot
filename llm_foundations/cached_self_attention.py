"""Lưu KV Cache của 1 head attention và tính toán với Query mới."""
import torch
from torch import nn

from llm_foundations.scaled_attention import causal_attention
from llm_foundations.self_attention import CausalSelfAttention


class CachedCausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int, 
        d_k: int,
        d_v: int
    ) -> None:
        """Khai báo các biến đầu vào của 1 head attention.
        
        Args:
            d_model: Kích thước đặc trưng của đầu vào.
            d_k: Kích thước đặc trưng của ma trận Key.
            d_v: kích thước đặc trưng của ma trận Value.
        """
        super().__init__()
        self.attention_head = CausalSelfAttention(d_model, d_k, d_v)

    def prefill(
        self,
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Tính causal attention cho prefix và trả K/V đã chiếu để cache.
        
        Args:
            x: Token của input prompt để tính trước attention output.
        """
        Q, K, V = x@self.attention_head.w_q, x@self.attention_head.w_k, x@self.attention_head.w_v

        output, attention_weights = causal_attention(Q, K, V)

        prefill_head = (output, attention_weights, (K, V))

        return prefill_head

    def decode(
        self,
        x_new: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Ứng với token đã sinh, tính attention output của 1 head với KV cache đã lưu.
        
        Args:
            x_new: Token mới sinh ra và embedding, rồi tính bộ Q,K,V của chính nó.
            cache: Các bộ Key, Value đã lưu trong cache.
        """
        Q_new, K_new, V_new = x_new@self.attention_head.w_q, x_new@self.attention_head.w_k, x_new@self.attention_head.w_v # Chiếu hidden state của vị trí mới thành Q/K/V
        K_old, V_old = cache # Lấy KV cache đã tính từ trước

        # Append KV Cache với K,V của token mới
        K_updated = torch.cat((K_old, K_new), dim=0)
        V_updated = torch.cat((V_old, V_new), dim=0)

        output, attention_weights = causal_attention(Q_new, K_updated, V_updated) # Query mới sẽ tính attention với bộ Key cache, 
                                                                                  # và tính ra attention output với bộ Value cache (gồm prompt prefill và cả chính token vừa sinh)

        decode_head = (output, attention_weights, (K_updated, V_updated))

        return decode_head
