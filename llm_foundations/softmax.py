"""Stable softmax implementation."""
import torch


def stable_softmax(logits: torch.Tensor) -> torch.Tensor:
    """Chuẩn hóa 2D logits theo dimension cuối."""
    row_max = logits.amax(dim=-1, keepdim=True) # Ứng với logits shape [B,V] -> row_max có shape là [B,1]

    shifted = logits - row_max # Broadcast: [B,v] - [B,1] = [B,v]
                               # Chuẩn hóa về logits luôn <= 0

    stable_exp = torch.exp(shifted) # Ứng với e của từng số, vẫn giữ nguyên shape [B,V]

    denominator_sum = stable_exp.sum(dim=-1, keepdim=True) # Tổng của từng expoential gộp theo cột, ứng với mỗi hàng 
                                                           # -> Ứng với mỗi hàng có shape là [B, 1]. Đây là mẫu của softmax

    softmax_probabilities = stable_exp / denominator_sum # Softmax shape: [B, V] trên tử / [B,1] mẫu -> broadcast shape thành [B,v]
                                           # Chuẩn hóa về xác suất thuộc khoảng [0,1] của softmax
    return softmax_probabilities
