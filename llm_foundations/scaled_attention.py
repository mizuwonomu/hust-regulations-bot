"""Tạo scaled dot-product attention với causal mask."""
import os
import sys

sys.path.append(os.path.abspath('.'))

import math

import torch

from llm_foundations.causal_mask import build_causal_mask
from llm_foundations.softmax import stable_softmax


def causal_attention(q: torch.Tensor, k: torch.Tensor, val: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Tạo output với từng cặp Q, K score, qua softmax và tính attention score
    với V. Ứng với cặp query ở sau cặp query đang xét, sử dụng causal mask 
    để các token sau không đóng góp vào output.

    Args:
    - q: Query của token đang được xét.
    - k: Key của token đang được xét độ liên quan với q.
    - val: Value đóng góp thông tin thực sự được trộn vào output.
    """
    scores = q@k.T # q: Shape [T, d_k] - mỗi query với chiều key
                   # k: Shape [T, d_k] -> Dot product tạo ra logits - trọng số của mỗi cặp (query, key)

    scaled_scores = scores / (math.sqrt(k.shape[1]))

    causal_bool = build_causal_mask(q.shape[0], device=q.device)

    masked_scores = scaled_scores.masked_fill(causal_bool, -torch.inf) # Ứng với từng token tương lai tại vị trí q đang xét,
                                                                       # gán trọng số -inf cho từng vị trí True của mask

    weights = stable_softmax(masked_scores) # Softmax chạy theo chiều keys

    output = weights@val 

    return output, weights
