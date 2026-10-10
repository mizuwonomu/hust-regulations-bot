"""Embedding input qua Tokenizer, rồi lấy phép cộng giữa Postion và Embedding Weights."""
import torch
from torch import nn

from llm_foundations.embedding import embedding_lookup


class InputEmbedding(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        max_seq_len: int,
        d_model: int
    ) -> None:
        """Khai báo các tham số đầu vào của input.
        
        Args:
            vocab_size: Số token có trong bộ vocab.
            max_seq_len: Độ dài tối đa của chuỗi text đầu vào.
            d_model: Số chiều đặc trưng của từng weights tương ứng.
        """
        super().__init__()
        # Khởi tạo 2 tham số học gồm Token Embedding, Position Embedding
        self.token_embed = nn.Parameter(torch.normal(0.0, 0.02, size=(vocab_size, d_model)))
        self.pos_embed = nn.Parameter(torch.normal(0.0, 0.02, size=(max_seq_len, d_model)))

    def forward(
        self,
        token_ids: torch.Tensor
    ) -> torch.Tensor:
        """Nhận chuỗi token IDs đã encode từ text, tra bảng embedding tương ứng,
        cộng 2 thành phần gồm Token Embedding, Position Embedding để thành Input Embedding.
        
        Args:
            token_ids: Chuỗi token IDs đã encode từ văn bản qua tokenizer.
        """
        if token_ids.ndim != 1:
            raise ValueError(
                f"token_ids must have rank 1, got rank {token_ids.ndim}"
            )

        seq_len = token_ids.shape[0]
        max_seq_len = self.pos_embed.shape[0]
        if seq_len > max_seq_len:
            raise ValueError(
                f"token_ids length {seq_len} exceeds max_seq_len {max_seq_len}"
            )

        vocab_size = self.token_embed.shape[0]
        if token_ids.numel() > 0 and (
            torch.any(token_ids < 0) or torch.any(token_ids >= vocab_size)
        ):
            raise ValueError(
                f"token_ids must be in the range [0, {vocab_size})"
            )

        token_embed_lookup = embedding_lookup(token_ids, self.token_embed)

        position_embed_lookup = self.pos_embed[:seq_len, :] # Lấy cho đến hết T tokens của chuỗi IDs, và lấy toàn bộ features của pos embed

        input_embed = token_embed_lookup + position_embed_lookup

        return input_embed
