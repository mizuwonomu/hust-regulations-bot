"""Character Tokenizer cho từng input text."""
import json


class CharacterTokenizer:
    def __init__(
        self,
        vocab_path: str
    ) -> None:
        """Nhận path chứa vocab, lập 2 dict encode và decode.
        
        Args:
            vocab_path: Path chứa tập vocabulary.
        """
        with open(vocab_path, encoding="utf-8") as vocab_file:
            self.vocab = json.load(vocab_file)
        if not isinstance(self.vocab, list):
            raise TypeError("vocab must be a list")
        if not self.vocab:
            raise ValueError("vocab must not be empty")
        if any(not isinstance(token, str) for token in self.vocab):
            raise TypeError("vocab tokens must be strings")
        if any(len(token) != 1 for token in self.vocab):
            raise ValueError("vocab tokens must be single characters")
        if len(self.vocab) != len(set(self.vocab)):
            raise ValueError("vocab tokens must be unique")

        self.char_to_id = {char: i for i, char in enumerate(self.vocab)} # Map char tới từng index IDs trong vocab
        self.id_to_char = {i: char for i, char in enumerate(self.vocab)} # Map index IDs từ các char trong vocab

    def encode(
        self,
        text: str
    ) -> list[int]:
        """Encode từng char xuất hiện trong text sang từng index tương ứng của vocabulary.
        
        Args:
            text: Chuỗi văn bản đầu vào.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if any(char not in self.char_to_id for char in text):
            raise ValueError("text contains unknown characters")
        return [self.char_to_id[char] for char in text]


    def decode(
        self,
        token_ids: list[int]
    ) -> str:
        """Decode từng index IDs sang chuỗi văn bản tương ứng của vocabulary.
                
        Args:
            token_ids: Chuỗi index của từng token IDs.
        """
        if not isinstance(token_ids, list):
            raise TypeError("token_ids must be a list")
        if any(not isinstance(token_id, int) or isinstance(token_id, bool) for token_id in token_ids):
            raise TypeError("token IDs must be integers")
        if any(token_id < 0 or token_id >= len(self.vocab) for token_id in token_ids):
            raise ValueError("token ID is out of range")
        return "".join(self.id_to_char[token_id] for token_id in token_ids)
