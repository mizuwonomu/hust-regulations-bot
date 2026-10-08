"""Một decoder block pre-norm gồm causal attention và FFN với hai nhánh residual."""
import torch  # noqa: I001
from torch import nn

from llm_foundations.layer_norm import LayerNorm
from llm_foundations.pre_norm_attention_residual import PreNormAttentionResidual
from llm_foundations.ffn import PositionWiseFFN


class PreNormDecoderBlock(nn.Module):
    """Biến đổi một chuỗi biểu diễn qua attention và FFN với hai LayerNorm độc lập."""
    def __init__(
        self, 
        d_model: int, 
        num_heads: int,
        d_k: int,
        d_v: int,
        d_ff: int,
        eps: float = 1e-5
    ) -> None:
        """Khởi tạo các sublayers và parameters dùng lại qua mọi lần forward.

        Args:
            d_model: Số features của input và output.
            num_heads: Số causal attention heads.
            d_k: Số features của query và key trong mỗi head.
            d_v: Số features của value trong mỗi head.
            d_ff: Số features trung gian của FFN.
            eps: Số dương dùng trong cả hai LayerNorm.
        """
        super().__init__()
        self.attention_residual = PreNormAttentionResidual(d_model, num_heads, d_k, d_v, eps)
        self.ffn_norm = LayerNorm(d_model, eps)
        self.ffn = PositionWiseFFN(d_model, d_ff)

    def forward(
        self,
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tính H = X + MHA(LN1(X)), rồi Y = H + FFN(LN2(H)).

        Args:
            x: Tensor [T, d_model], cùng dtype và device với parameters.

        Returns:
            Output [T, d_model] và attention weights [num_heads, T, T].
        """
        attention_residual, attention_weights = self.attention_residual(x) # Sublayer attention đã bao gồm phép cộng với input gốc

        attention_normalized = self.ffn_norm(attention_residual)

        mlp_result = self.ffn(attention_normalized)

        mlp_residual = attention_residual + mlp_result

        return mlp_residual, attention_weights
