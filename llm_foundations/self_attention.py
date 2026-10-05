"""Forward propagation từ đầu vào, qua attention và tính ra output."""
import os
import sys

sys.path.append(os.path.abspath('.'))

import torch
from torch import nn

from llm_foundations.scaled_attention import causal_attention


class CausalSelfAttention(nn.Module):
    """Khai báo module cho single-head self-attention."""

    def __init__(
        self, 
        d_model: int, 
        d_k: int,
        d_v: int
    ) -> None:
        super().__init__()
        # Tạo các weights một lần cho mỗi instance, gán thành thuộc tính Module
        self.w_q = nn.Parameter(nn.init.xavier_uniform_(torch.empty(d_model, d_k)))
        self.w_k = nn.Parameter(nn.init.xavier_uniform_(torch.empty(d_model, d_k)))
        self.w_v = nn.Parameter(nn.init.xavier_uniform_(torch.empty(d_model, d_v)))

    def forward(
        self,
        X: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Thực hiện forward propagation.

        Với input cho trước, tiến hành nhân các ma trận input với 3 ma trận weights Q,K và V
        và ra các Attention weights, rồi nhân với V để ra output cuối.
        """

        Q, K, V = X@self.w_q, X@self.w_k, X@self.w_v

        output, attention_weights = causal_attention(Q, K, V)

        return output, attention_weights
    