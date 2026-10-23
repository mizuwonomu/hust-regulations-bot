"""Ghép decoder-only LM với KV cache cho prefill và decode từng token."""
import torch  # noqa: I001
from torch import nn

from llm_foundations.embedding import embedding_lookup
from llm_foundations.embeddings.input_embedding import InputEmbedding
from llm_foundations.layer_norm import LayerNorm
from llm_foundations.kv_cache.decoder_block import CachedPreNormDecoderBlock
from llm_foundations.lm_head import LMHead


class CachedDecoderOnlyLM(nn.Module):
    """Tính logits với các block độc lập và cache do bên gọi quản lý."""
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
        """Khởi tạo embedding, các decoder block, final norm và LM head dùng chung hai pha.

        Args:
            vocab_size: Số token trong vocabulary và số cột logits đầu ra.
            max_seq_len: Số vị trí tối đa trong bảng position embedding.
            d_model: Số features của embedding và hidden states.
            num_layers: Số decoder block độc lập xếp nối tiếp.
            num_heads: Số attention heads trong mỗi block.
            d_k: Số features của query và key trong mỗi head.
            d_v: Số features của value trong mỗi head.
            d_ff: Số features trung gian của FFN.
            eps: Hằng số ổn định cho hai LayerNorm mỗi block và final norm.
        """
        super().__init__()
        self.input_embedding = InputEmbedding(vocab_size, max_seq_len, d_model)
        self.blocks = nn.ModuleList([CachedPreNormDecoderBlock(d_model, num_heads, d_k, d_v, d_ff, eps) for _ in range(num_layers)])
        self.final_norm = LayerNorm(d_model, eps)
        self.lm_head = LMHead(d_model, vocab_size)


    def prefill(
        self,
        token_ids: torch.Tensor
    ) -> tuple[torch.Tensor, list[list[tuple[torch.Tensor, torch.Tensor]]]]:
        """Tính logits [T, V] của prefix và tạo cache theo thứ tự layer/head.

        Args:
            token_ids: IDs long [T] của một chuỗi, cùng device với model,
                có độ dài từ 1 đến sức chứa bảng position embedding.
        """
        layers_kv_caches: list[list[tuple[torch.Tensor, torch.Tensor]]] = []

        input_embed = self.input_embedding(token_ids)
        hidden_states = input_embed

        for block in self.blocks:
            mlp_residual, _, kv_caches = block.prefill(hidden_states)

            layers_kv_caches.append(kv_caches)

            hidden_states = mlp_residual

        last_hidden_state = self.final_norm(hidden_states)

        logits = self.lm_head(last_hidden_state)

        return logits, layers_kv_caches

    def decode(
        self,
        token_ids_new: torch.Tensor,
        caches: list[list[tuple[torch.Tensor, torch.Tensor]]]
    ) -> tuple[torch.Tensor, list[list[tuple[torch.Tensor, torch.Tensor]]]]:
        """Tính logits [1, V] tại position T và trả cache đã nối thêm một token.

        Args:
            token_ids_new: IDs long [1] của token mới, cùng device với model.
            caches: Cache hợp lệ ứng với prefix và parameters hiện tại,
                tổ chức theo caches[layer][head] = (K, V). Mỗi K có shape
                [T, d_k], mỗi V có shape [T, d_v], cùng dtype/device với model;
                T bằng nhau ở mọi head/layer và còn chỗ cho một position mới.
        """
        updated_layers_kv_caches: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
        prefix_len = caches[0][0][0].shape[0] # Lần lượt từ trái sang, lấy cache của layer đầu
                                              # -> Lấy tuple (K, V) của head đầu
                                              # -> Rồi lấy tensor K, shape [T, d_k] để ra độ dài prefix T

        new_token_embed_lookup = embedding_lookup(token_ids_new, self.input_embedding.token_embed)
        new_pos_embed_lookup = self.input_embedding.pos_embed[prefix_len, :] # Lấy chính position index bằng độ dài cache trước append

        new_input_embed = new_token_embed_lookup + new_pos_embed_lookup
        new_input = new_input_embed

        for block, cache in zip(self.blocks, caches): # Duyệt theo cặp tương ứng: (block, cache layer)
            new_mlp_residual, _, updated_caches = block.decode(new_input, cache)

            updated_layers_kv_caches.append(updated_caches)

            new_input = new_mlp_residual

        new_last_hidden_state = self.final_norm(new_input)

        new_logits = self.lm_head(new_last_hidden_state)

        return new_logits, updated_layers_kv_caches
