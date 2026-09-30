"""Cross-entropy loss cho từng logits."""
import torch


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor, reduction) -> torch.Tensor:
    """Tính loss cross entropy và gộp lại."""
    row_maximum = logits.amax(dim=-1, keepdim=True)

    shifted = logits - row_maximum

    stable_exp = torch.exp(shifted)

    denominator_sum = stable_exp.sum(dim=-1, keepdim=True)

    target_column = targets.unsqueeze(1) # Từ shape của target - index của token đúng [B], biến đổi thành shape [B,1]
    selected_logits = logits.gather(dim=1, index=target_column) # Ứng với mỗi cột của logits, chọn ra logit tương ứng với index của token đúng
    selected_logits = selected_logits.squeeze(1) # Trở về shape [B]

    # Squeeze max, deno sum về shape [B]
    row_maximum = row_maximum.squeeze(1)
    denominator_sum = denominator_sum.squeeze(1)
    per_sample_loss = (row_maximum - selected_logits) + torch.log(denominator_sum) # L = (m-z_k + log(sum(e^shifted)))

    if (reduction == "none"): # Giữ nguyên từng loss của từng mẫu
        return per_sample_loss

    if (reduction == "sum"): # Gộp loss theo tổng và trả về scalar - tối thiểu hóa tổng loss
        per_sample_loss = per_sample_loss.sum()
        return per_sample_loss

    if (reduction == "mean"):
        per_sample_loss = per_sample_loss.mean()
        return per_sample_loss