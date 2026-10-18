"""Sinh token tự hồi quy bằng cách gọi model trên toàn bộ prefix."""

from collections import abc

import torch

from llm_foundations.decoder_only_lm import DecoderOnlyLM
from llm_foundations.decoding import greedy_token


def generate(
    model: DecoderOnlyLM,
    token_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    selector: abc.Callable[[torch.Tensor], torch.Tensor] = greedy_token,
) -> torch.Tensor:
    module_modes = [(module, module.training) for module in model.modules()]

    try:
        model.eval()
        count_new_tokens = 0

        with torch.no_grad():
            while (count_new_tokens < max_new_tokens 
                and token_ids.shape[0] < model.input_embedding.pos_embed.shape[0]
                ): # T tokens chuỗi sinh hiện tại không vượt quá max độ dài position
                logits = model(token_ids) # Compute logits

                next_token = selector(logits[-1, :]) # Sampling hoặc greedy token để chọn ra next token 
                                                    # lấy tensor logits hàng cuối [V]
                next_token = next_token.unsqueeze(0)

                count_new_tokens += 1

                append_next_token = torch.cat((token_ids, next_token), dim=0) # Nối chuỗi input token IDs với token vừa được sinh

                token_ids = append_next_token

        return token_ids
    finally:
        for module, training in module_modes:
            module.train(training)