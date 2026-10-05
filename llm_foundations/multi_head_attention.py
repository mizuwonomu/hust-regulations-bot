"""Multi-head causal self-attention với các head độc lập và output projection."""
import torch
from torch import nn

from llm_foundations.self_attention import CausalSelfAttention


class MultiHeadCausalSelfAttention(nn.Module):
    """Khởi tạo các head độc lập và output projection học được.

    Args:
        - d_model: Số features của input và output cuối.
        - num_heads: Số attention heads độc lập.
        - d_k: Số features của query và key trong mỗi head.
        - d_v: Số features của value và output trong mỗi head.
    """
    def __init__(
        self, 
        d_model: int,
        num_heads: int,
        d_k: int,
        d_v: int
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList([CausalSelfAttention(d_model, d_k, d_v) for _ in range(num_heads)]) # Tạo các bộ Q/K/V weights riêng của từng head
        self.w_o = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_heads*d_v, d_model))) # Khai báo weight riêng của output

    def forward(
        self,
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tính output và attention weights.

        Mỗi head nhận cùng input; outputs được ghép theo chiều feature rồi
        chiếu qua w_o. Attention weights giữ nguyên thứ tự các head.

        Args:
            - x: Một chuỗi [T, d_model], cùng dtype và device với parameters.
        """
        outputs = [] 
        weights = []
        for head in self.heads:
            output_h, weights_h = head(x)
            outputs.append(output_h)
            weights.append(weights_h)

        concat_outputs = torch.concat(outputs, dim=1) # Concat theo shape [T, num_heads * d_v]
        full_head_weights = torch.stack(weights, dim=0) # Stack theo shape [num_heads, T, T]

        output = concat_outputs@self.w_o

        return output, full_head_weights
    