"""Embedding lookup cho từng token IDs."""
import torch


def embedding_lookup(token_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    output = weight[token_ids, :] # Advanced indexing: Token IDs có shape [B, T], weight có shape [V, D]
                                    # Ứng với từng token cần tìm của từng batch, tra index tương ứng với V của weight, độ dài D
                                    # token_ids: shape [B,T] của index thành prefix shape output, : để lấy toàn bộ độ dài của vector feature D

    return output
