import torch  # noqa: I001
from torch import nn


from llm_foundations.embeddings.input_embedding import InputEmbedding
from llm_foundations.decoder_block import PreNormDecoderBlock
from llm_foundations.layer_norm import LayerNorm
from llm_foundations.lm_head import LMHead


class DecoderOnlyLM(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_k: int,
        d_v: int,
        d_ff: int,
        eps: float = 1e-5
    ) -> None:
        super().__init__()
        self.input_embedding = InputEmbedding(vocab_size, max_seq_len, d_model)
        self.blocks = nn.ModuleList([PreNormDecoderBlock(d_model, num_heads, d_k, d_v, d_ff, eps) for _ in range(num_layers)])
        self.final_norm = LayerNorm(d_model, eps)
        self.lm_head = LMHead(d_model, vocab_size)

    def forward(
        self,
        token_ids: torch.Tensor
    ) -> torch.Tensor:
        input_embed = self.input_embedding(token_ids)

        input = input_embed
        for block in self.blocks:
            mlp_residual, _ = block(input) # Tính hidden states của từng transformer block

            input = mlp_residual # Output residual làm input để đưa nối tiếp vào block transformer sau

        last_hidden_state = self.final_norm(mlp_residual)

        logits = self.lm_head(last_hidden_state)

        return logits
