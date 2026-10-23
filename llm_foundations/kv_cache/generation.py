"""Sinh token tự hồi quy với một lần prefill và KV cache dùng lại khi decode."""
from collections import abc # noqa: I001

import torch

from llm_foundations.kv_cache.decoder_only_lm import CachedDecoderOnlyLM
from llm_foundations.decoding import greedy_token


def generation(
    model: CachedDecoderOnlyLM,
    token_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    selector: abc.Callable[[torch.Tensor], torch.Tensor] = greedy_token,
) -> torch.Tensor:
    """Sinh IDs trong hai budget và khôi phục trạng thái suy luận của bên gọi.

    Args:
        model: Model có prefill, decode và bảng position cố định.
        token_ids: Prompt long [P], cùng device với model.
        max_new_tokens: Số token mới tối đa được sinh.
        selector: Hàm nhận logits [V] và trả scalar ID long cùng device.
    """
    max_seq_len = model.input_embedding.pos_embed.shape[0]
    if max_new_tokens == 0 or token_ids.shape[0] >= max_seq_len:
        return token_ids

    module_modes = [(module, module.training) for module in model.modules()]

    try:
        model.eval()

        with torch.no_grad():
            logits, layer_kv_caches = model.prefill(token_ids)

            count_new_tokens = 0

            while (
                count_new_tokens < max_new_tokens
                and token_ids.shape[0] < max_seq_len
            ):
                next_token = selector(logits[-1, :])

                next_token = next_token.unsqueeze(0)

                count_new_tokens += 1

                append_next_token = torch.cat((token_ids, next_token), dim=0)

                token_ids = append_next_token

                if count_new_tokens >= max_new_tokens or token_ids.shape[0] >= max_seq_len:
                    break

                # Decode token vừa chọn và giữ cache mới cho vòng sau
                logits, layer_kv_caches = model.decode(next_token, layer_kv_caches)

        return token_ids
    finally:
        for module, training in module_modes:
            module.train(training)
