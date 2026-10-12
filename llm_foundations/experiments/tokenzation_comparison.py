"""So sánh token, ID, số lượng và decode của character, byte và BPE toy."""
import json
import os
import sys

sys.path.append(os.path.abspath('.'))

from tokenizers import Tokenizer  # Điều phối training
from tokenizers.decoders import Fuse  # Nối tokens thành các chuỗi
from tokenizers.models import BPE  # Model lưu vocab
from tokenizers.trainers import BpeTrainer  # Học vocabulary/rules từ corpus

from llm_foundations.config import CHARACTER_VOCAB_PATH
from llm_foundations.tokenizer.character_tokenizer import CharacterTokenizer

texts = ["ababa", "ếế"]
character_tokenizer = CharacterTokenizer(CHARACTER_VOCAB_PATH)
bpe_training_text = "abababab"

bpe_model = BPE()
bpe_tokenizer = Tokenizer(bpe_model)
bpe_tokenizer.decoder = Fuse()
bpe_trainer = BpeTrainer(
    initial_alphabet=["a", "b", "ế"],
    vocab_size=5, # Định nghĩa lượng vocab giống manual, a,b,ế,ab,abab
    min_frequency=1, # Cặp token phải xuất hiện ít nhất một lần trong corpus ở trạng thái training hiện tại
    special_tokens=[],
    show_progress=True,
)
bpe_tokenizer.train_from_iterator([bpe_training_text], bpe_trainer) # Đếm cặp token liền nhau, chọn cặp có tần suất cao nhất, merge rồi cập nhật tần suất

bpe_vocab = bpe_tokenizer.get_vocab()
bpe_merges = json.loads(bpe_tokenizer.to_str())["model"]["merges"]
print(f"BPE toy - trained vocabulary: {bpe_vocab}")
print(f"BPE toy - trained merge rules: {bpe_merges}")

bpe_id_to_token = {
    0: "a",
    1: "b",
    2: "ab",
    3: "abab",
} # Vocabulary toy cho BPE
bpe_merge_rules = [((0, 1), 2), ((2, 2), 3)]

bpe_training_trace = [
    [0, 1, 0, 1, 0, 1, 0, 1],
    [2, 2, 2, 2],
    [3, 3],
] # Khai báo từng bước học merge, từ tách từng char -> ghép cặp (0,1) ra 2, và từ cặp (2,2) ra 3
bpe_encode_trace = [
    [0, 1, 0, 1, 0],
    [2, 2, 0],
    [3, 0],
] # Encode chuỗi text ababa

print(f"BPE toy - manual rules: {bpe_merge_rules}")
print(
    f"BPE toy - manual training example text={bpe_training_text!r}: "
    f"{bpe_training_trace[0]} -> {bpe_training_trace[1]} -> {bpe_training_trace[2]}"
)
print(
    "BPE toy - manual encode example text='ababa': "
    f"{bpe_encode_trace[0]} -> {bpe_encode_trace[1]} -> {bpe_encode_trace[2]}"
)

for text in texts:
    character_ids = character_tokenizer.encode(text)
    character_tokens = [
        character_tokenizer.id_to_char[token_id] for token_id in character_ids
    ]
    character_decoded = character_tokenizer.decode(character_ids)
    print(
        f"character | text={text!r} | tokens={character_tokens} "
        f"| ids={character_ids} | count={len(character_ids)} "
        f"| decoded={character_decoded!r} "
        f"| equality={character_decoded == text}"
    )

    byte_ids = list(text.encode(encoding="utf-8"))
    byte_values = byte_ids
    byte_decoded = bytes(byte_ids).decode(encoding="utf-8")
    print(
        f"byte | text={text!r} | byte_values={byte_values} "
        f"| ids={byte_ids} | count={len(byte_ids)} "
        f"| decoded={byte_decoded!r} | equality={byte_decoded == text}"
    )

    bpe_encoding = bpe_tokenizer.encode(text)
    bpe_decoded = bpe_tokenizer.decode(bpe_encoding.ids)

    print(
        f"BPE toy - model | text={text!r} | tokens={bpe_encoding.tokens} "
        f"| ids={bpe_encoding.ids} | count={len(bpe_encoding.ids)} "
        f"| decoded={bpe_decoded!r} | equality={bpe_decoded == text}"
    )

    if text == "ababa":
        bpe_ids = bpe_encode_trace[-1]
        bpe_tokens = [bpe_id_to_token[token_id] for token_id in bpe_ids]
        bpe_decoded = "".join(bpe_tokens)
        print(
            f"BPE toy - manual | text={text!r} | tokens={bpe_tokens} "
            f"| ids={bpe_ids} | count={len(bpe_ids)} "
            f"| decoded={bpe_decoded!r} | equality={bpe_decoded == text}"
        )

    else:
        print(
            f"BPE toy - manual | text={text!r} | tokens=unsupported "
            "| ids=unsupported | count=None | decoded=None | equality=None"
        )
