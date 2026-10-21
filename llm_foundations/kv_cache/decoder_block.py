"""Tính decoder block với Multi-head Attention có KV Cache và qua lớp MLP."""
import torch  # noqa: I001
from torch import nn

from llm_foundations.layer_norm import LayerNorm
from llm_foundations.kv_cache.multi_head_attention import CachedMultiHeadAttention
from llm_foundations.ffn import PositionWiseFFN


class CachedPreNormDecoderBlock(nn.Module):
    def __init__(
        self, 
        d_model: int,
        num_heads: int,
        d_k: int,
        d_v: int,
        d_ff: int,
        eps: float = 1e-5
    ) -> None:
        """Khai báo các tham số cho Decoder Block.

        Args:
            d_model: Kích thước của đặc trưng đầu vào.
            num_heads: Số heads trong 1 transformer block.
            d_k: Kích thước đặc trưng ma trận Key.
            d_v: Kích thước đặc trưng ma trận Value.
            d_ff: Kích thước đặc trưng ma trận MLP Weights.
            eps: Constant chuẩn hóa đầu vào.
        """
        super().__init__()
        self.normalize_input = LayerNorm(d_model, eps)
        self.cached_mha = CachedMultiHeadAttention(d_model, num_heads, d_k, d_v)
        self.ffn_norm = LayerNorm(d_model, eps)
        self.ffn = PositionWiseFFN(d_model, d_ff)

    def prefill(
        self,
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Tính toán trước cho Decoder Block với input prompt.

        Args: 
            x: Hidden states để đi vào decoder block.
        
        Returns:
            mlp_residual: Phép cộng giữa hidden state và kết quả linear MLP.
            full_head_weights: Các trọng số attention heads.
            kv_caches: List các tuple KV Caches đã lưu của từng heads.
        """
        normalized_input = self.normalize_input(x)
        output, full_head_weights, kv_caches = self.cached_mha.prefill(normalized_input)

        attention_residual = output + x

        attention_normalized = self.ffn_norm(attention_residual)

        mlp_result = self.ffn(attention_normalized)

        mlp_residual = mlp_result + attention_residual

        return mlp_residual, full_head_weights, kv_caches

    def decode(
        self,
        x_new: torch.Tensor,
        caches: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Tính toán Decoder Block cho các token mới sinh.

        Args:
            x_new: Token mới sinh và đi vào chuỗi tính toán.
            caches: List các KV Caches đã lưu trước đó của các heads.

        Returns:
            new_mlp_residual: Phép cộng giữa hidden state và kết quả linear MLP của token mới.
            new_full_head_weights: Các trọng số attention heads khi tính toán token mới.
            updated_caches: List KV Caches khi append token mới.
        """
        normalized_input = self.normalize_input(x_new)
        new_output, full_new_head_weights, updated_caches = self.cached_mha.decode(normalized_input, caches)

        new_attention_residual = new_output + x_new

        new_attention_normalized = self.ffn_norm(new_attention_residual)

        new_mlp_result = self.ffn(new_attention_normalized)

        new_mlp_residual = new_mlp_result + new_attention_residual

        return new_mlp_residual, full_new_head_weights, updated_caches
