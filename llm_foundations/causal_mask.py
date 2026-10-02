"""Tạo boolean mask đánh dấu các token tương lai cần mask."""

import torch


def build_causal_mask(seq_len: int, *, device: torch.device | str = "cpu") -> torch.Tensor:
  """Tạo matrix từ độ dài chuỗi, gán boolean mask ứng với các token.

  Args:
  - seq_len: Độ dài chuỗi đang xét -> Chuyển thành matrix shape [seq_len, seq_len]
  - device: Khai báo CPU hoặc CUDA.
  """
  col = torch.arange(start=0, end=seq_len, device=device) # Tạo cột cho số seq len tokens

  col = col.unsqueeze(0) # Tạo shape [1, seq_len]

  row = torch.arange(start=0, end=seq_len, device=device)

  row = row.unsqueeze(1) # Tạo shape [seq_len, 1]
  
  mask = col > row # Broadcasting so sánh mọi cặp chỉ số cột j và hàng i, tạo mask [T,T].
                    # Hàng i đại diện query đang xét; cột j đại diện key. 
                    # Khi j > i, key nằm ở tương lai nên mask là True để chặn query nhìn thấy key đó

  return mask
