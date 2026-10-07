"""LayerNorm theo features của từng token với scale và shift học được."""
import torch
from torch import nn


class LayerNorm(nn.Module):
    """Chuẩn hóa từng token độc lập bằng các parameters dùng chung."""
    def __init__(
        self, 
        d_model: int,
        eps: float = 1e-5
    ) -> None:
        """Khởi tạo parameters và cấu hình chuẩn hóa.

        Args:
        - d_model: Số features của mỗi token.
        - eps: Số dương cộng vào variance trước khi lấy căn.
        """
        super().__init__()
        # Định nghĩa 2 hệ số học cho scale and shift
        self.gamma = nn.Parameter(torch.ones(d_model), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(d_model), requires_grad=True)

        self.d_model = d_model
        self.eps = eps
        

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Chuẩn hóa input rồi áp dụng scale và shift theo feature.

        Args:
        - x: Tensor [T, d_model], cùng dtype và device với parameters.
        """
        # Tính mean và population variance theo features của từng token
        mu = torch.sum(x, dim=1, keepdim=True) / self.d_model # Mean có shape [T, 1] để broadcast theo features
        var = torch.sum(torch.square(x - mu), dim=1, keepdim=True) / self.d_model # Tổng phương sai / d_model -> shape [T, 1]
 
        # Chuẩn hóa bằng mean và variance của từng token
        x_normalized = (x - mu) / (torch.sqrt(var + self.eps))

        # Áp dụng scale và shift học được theo từng feature
        normalized_output = self.gamma*x_normalized + self.beta

        return normalized_output
