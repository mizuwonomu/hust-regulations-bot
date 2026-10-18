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
        """Khai báo các component cần thiết để ghép, gồm:
        - input_embedding: Khối embedding input sau khi lookup IDs.
        - blocks: Các Pre-LN Decoder block tương ứng với các layers.
        - final_norm: Lớp Layer Norm sau khi tính xong hidden state của layer cuối.
        - lm_head: Lớp LM Head để tính toán logits với hidden state.
        

        Args:
            vocab_size: Kích thước - số từ vựng của vocabulary.
            max_seq_len: Độ dài tối đa của chuỗi đầu vào.
            d_model: Kích thước đặc trưng của đầu vào.
            num_layers: Số Transformer block chồng lên nhau.
            num_heads: Số attention heads có trong 1 layer.
            d_k: Kích thước đặc trưng của ma trận Key.
            d_v: Kích thước đặc trưng của ma trận Value.
            d_ff: Kích thước đặc trưng của ma trận lớp FFN.
            eps: Hằng số epsilon dùng để chuẩn hóa đầu vào.
        """
        super().__init__()
        self.input_embedding = InputEmbedding(vocab_size, max_seq_len, d_model)
        self.blocks = nn.ModuleList([PreNormDecoderBlock(d_model, num_heads, d_k, d_v, d_ff, eps) for _ in range(num_layers)])
        self.final_norm = LayerNorm(d_model, eps)
        self.lm_head = LMHead(d_model, vocab_size)

    def forward(
        self,
        token_ids: torch.Tensor
    ) -> torch.Tensor:
        """Luồng đi của full Transformer block.
        Bắt đầu từ chuỗi prompt đầu vào, lookup IDs, rồi embedding với Position Embedding và Token Embedding.
        Sau đó, input sẽ lần lượt đi qua các transformer block để tính ra output. 
        Output của một block sẽ là input của block sau nó. 
        Sau khi tính ra hidden state của block transformer cuối, sẽ đi qua một lớp chuẩn hóa cuối.
        Cuói cùng, tính toán ra logits giữa hidden state cuối và LM Head Weights.

        Args:
            token_ids: Chuỗi IDs của token đầu vào."""
        input_embed = self.input_embedding(token_ids)

        input = input_embed
        for block in self.blocks:
            mlp_residual, _ = block(input) # Tính hidden states của từng transformer block

            input = mlp_residual # Output residual làm input để đưa nối tiếp vào block transformer sau

        last_hidden_state = self.final_norm(mlp_residual)

        logits = self.lm_head(last_hidden_state)

        return logits
