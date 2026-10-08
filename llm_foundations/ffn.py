"""FFN xử lý từng token độc lập qua hai phép affine và activation GELU."""
import math

import torch
from torch import nn


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Tính GELU theo công thức erf cho từng phần tử, giữ nguyên shape.

    Args:
        x: Tensor số thực cần áp dụng activation.
    """
    v = (x / 2) * (1 + torch.erf(x / math.sqrt(2)))

    return v 

class PositionWiseFFN(nn.Module):
    """Biến đổi features của từng token bằng cùng một bộ parameters học được."""

    def __init__(
        self, 
        d_model: int,
        d_ff: int
    ) -> None:
        """Khởi tạo weights và biases cho hai phép affine.

        Args:
            d_model: Số features của input và output.
            d_ff: Số features trung gian trước và sau GELU.
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        # Đăng ký parameters một lần, dùng chung cho mọi vị trí token
        self.w_1 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(d_model, d_ff)))
        self.b_1 = nn.Parameter(torch.zeros(d_ff))
        self.w_2 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(d_ff, d_model)))
        self.b_2 = nn.Parameter(torch.zeros(d_model))

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Tính affine - GELU - affine độc lập trên từng hàng input.

        Args:
            x: Tensor [T, d_model], cùng dtype và device với parameters.

        Returns:
            Tensor [T, d_model], cùng dtype và device với x.
        """
        # Chiếu sang d_ff features, bias broadcast theo chiều token
        first_linear = x@self.w_1 + self.b_1
        activation = gelu(first_linear)

        # Chiếu về d_model, không thêm activation để cho phép output âm hoặc dương
        mlp_result = activation@self.w_2 + self.b_2

        return mlp_result
