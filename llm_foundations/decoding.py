"""Chọn token tiếp theo bằng greedy hoặc sampling, cho phép -inf làm mask."""

import math
import numbers

import torch

from llm_foundations.softmax import stable_softmax


def _validate_logits(logits: torch.Tensor) -> None:
    """Kiểm tra logits [V] dense floating và miền giá trị được hỗ trợ.

    Args:
        logits: Tensor [V], cho phép -inf làm mask nếu còn ít nhất một giá trị hữu hạn.
    """
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits phải là torch.Tensor")
    if logits.layout != torch.strided:
        raise TypeError("logits phải là tensor dense strided")
    if not torch.is_floating_point(logits):
        raise TypeError("logits phải có dtype floating")
    if logits.ndim != 1 or logits.numel() < 1:
        raise ValueError("logits phải có dạng [V] với V >= 1")
    if (
        torch.isnan(logits).any().item()
        or torch.isposinf(logits).any().item()
        or not torch.isfinite(logits).any().item()
    ):
        raise ValueError("logits chỉ được chứa giá trị hữu hạn hoặc -inf mask")


def _validate_generator(
    generator: torch.Generator | None, device: torch.device
) -> None:
    """Kiểm tra generator có kiểu và device phù hợp với logits.

    Args:
        generator: Generator tùy chọn dùng khi lấy mẫu.
        device: Device của logits cần khớp với generator.
    """
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("generator phải là torch.Generator hoặc None")
    if generator is not None and generator.device != device:
        raise ValueError("device của generator phải khớp với logits")


def greedy_token(logits: torch.Tensor) -> torch.Tensor:
    """Trả về scalar ID có logit lớn nhất từ logits [V].

    Args:
        logits: Tensor [V], cho phép -inf làm mask nếu còn ít nhất một giá trị hữu hạn.
    """
    _validate_logits(logits)
    next_token = logits.argmax()

    return next_token


def sampling(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None
) -> torch.Tensor:
    """Lấy mẫu một scalar ID từ logits [V] với tối đa một bộ lọc.

    Args:
        logits: Tensor [V], cho phép -inf làm mask nếu còn giá trị hữu hạn.
        temperature: Số thực hữu hạn lớn hơn 0, mặc định là 1.0.
        top_k: Int trong [1, V] hoặc None.
        top_p: Số thực hữu hạn trong (0, 1] hoặc None.
        generator: Generator cùng device với logits hoặc None.
    """
    _validate_logits(logits)

    if isinstance(temperature, bool) or not isinstance(temperature, numbers.Real):
        raise TypeError("temperature phải là số thực khác bool")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature phải hữu hạn và lớn hơn 0")
    if top_k is not None:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k phải là int khác bool hoặc None")
        if not 1 <= top_k <= logits.numel():
            raise ValueError("top_k phải nằm trong [1, V]")
    if top_p is not None:
        if isinstance(top_p, bool) or not isinstance(top_p, numbers.Real):
            raise TypeError("top_p phải là số thực khác bool hoặc None")
        if not math.isfinite(top_p) or not 0 < top_p <= 1:
            raise ValueError("top_p phải hữu hạn và nằm trong (0, 1]")
    if top_k is not None and top_p is not None:
        raise ValueError("Không thể chứa đồng thời top_k lẫn top_p")
    _validate_generator(generator, logits.device)

    # Chia logits trước khi tính softmax để điều chỉnh độ tập trung phân phối
    tmp_distributed_logits = logits / temperature
    tmp_distributed_prob = stable_softmax(tmp_distributed_logits)

    if top_k is not None:
        top_k_tokens = tmp_distributed_prob.topk(top_k)
        top_k_ids = top_k_tokens.indices

        selected_ids = torch.multinomial(
            top_k_tokens.values, num_samples=1, generator=generator
        )
        # Ánh xạ vị trí trong top-k về ID từ vựng ban đầu
        next_token = top_k_ids[selected_ids[0]]

        return next_token

    if top_p is not None:
        sorted_prob = tmp_distributed_prob.sort(descending=True)

        cumulative_prob = sorted_prob.values.cumsum(dim=0)
        prefix_indices = (cumulative_prob >= top_p).nonzero()
        first_reach = prefix_indices[0, 0]

        # Giữ cả token vượt ngưỡng tích lũy
        top_p_ids = sorted_prob.indices[:first_reach + 1]
        top_p_probs = sorted_prob.values[:first_reach + 1]

        # Chuẩn hóa lại tổng khối xác suất đã giữ
        normalized_p_probs = top_p_probs / torch.sum(top_p_probs)

        selected_ids = torch.multinomial(
            normalized_p_probs, num_samples=1, generator=generator
        )
        # Ánh xạ vị trí trong top-p về ID từ vựng ban đầu
        next_token = top_p_ids[selected_ids[0]]

        return next_token

    selected_ids = torch.multinomial(
        tmp_distributed_prob, num_samples=1, generator=generator
    )
    # Với vector đầy đủ, vị trí lấy mẫu chính là ID từ vựng ban đầu
    next_token = selected_ids[0]

    return next_token
