"""Sublayer causal self-attention pre-norm với kết nối residual.

Chuẩn hóa từng token trước attention, rồi cộng output đã chiếu vào input gốc.
Xử lý một sequence không batch và trả về cả output lẫn attention probabilities.
"""
import torch
from torch import nn

from llm_foundations.layer_norm import LayerNorm
from llm_foundations.multi_head_attention import MultiHeadCausalSelfAttention


class PreNormAttentionResidual(nn.Module):
    """Kết hợp LayerNorm và multi-head causal attention với nhánh skip input gốc."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_k: int,
        d_v: int,
        eps: float = 1e-5
    ) -> None:
        """Khởi tạo LayerNorm và multi-head attention dùng lại qua các lần forward.

        Args:
            d_model: Số features của input và output cuối, là số nguyên dương.
            num_heads: Số attention heads độc lập, là số nguyên dương.
            d_k: Số features của query và key trong mỗi head, là số nguyên dương.
            d_v: Số features của value và output mỗi head, là số nguyên dương.
            eps: Số thực dương cộng vào population variance trước khi lấy căn.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.eps = eps

        # Đăng ký các child modules một lần để dùng chung parameters qua các lần forward
        self.multi_head_attention = MultiHeadCausalSelfAttention(self.d_model, self.num_heads, self.d_k, self.d_v)
        self.normalized_input = LayerNorm(self.d_model, self.eps)

    def forward(
        self,
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tính Y = X + MHA(LN(X)) và trả về attention probabilities.

        Args:
            x: Tensor float32 hoặc float64 hữu hạn, shape [T, d_model] với T > 0,
                biểu diễn một sequence không batch, cùng dtype và device với parameters.

        Returns:
            Tuple (output_residual, attention_weights), cùng dtype và device với x.
            output_residual có shape [T, d_model], là input gốc cộng attention output.
            attention_weights có shape [num_heads, T, T], theo thứ tự head, query, key;
            là probabilities với các vị trí key tương lai bằng 0.
        """
        # Chuẩn hóa từng token trước khi các head tính query, key và value
        pre_ln_input_attention = self.normalized_input(x)

        # Lấy attention output sau khi ghép các head và chiếu qua W_O
        output, attention_weights = self.multi_head_attention(pre_ln_input_attention)

        # Cộng vào input gốc để giữ nhánh skip chưa chuẩn hóa
        output_residual = x + output

        return output_residual, attention_weights
