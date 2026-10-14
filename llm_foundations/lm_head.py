"""Lớp LM Head projection để dự đoán ra logits, cuối cùng là
xác suất token được chọn."""
import torch
from torch import nn


class LMHead(nn.Module):
    def __init__(
        self, 
        d_model: int,
        vocab_size: int
    ) -> None:
        """Khởi tạo tham số học của lớp LM Head - dùng trong phép
        tính logits của token tiếp theo.
        
        Args:
            d_model: Chiều đặc trưng của mô hình.
            vocab_size: Số token có trong vocabulary."""
        super().__init__()
        self.w_lmhead = nn.Parameter(nn.init.xavier_uniform_(torch.empty(d_model, vocab_size)))

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Nhận hidden state của token ở thời điểm hiện tại, tính toán 
        với LM Head Weights và output ra logits.
        """
        logits = hidden_states@self.w_lmhead

        return logits
