"""Kiểm tra hợp đồng tokenizer ký tự trên vocabulary tạm thời."""

import builtins
import inspect
import itertools
import json
import os
from pathlib import Path

import pytest

from llm_foundations.tokenizer.character_tokenizer import CharacterTokenizer


_VOCAB_A = ("b", " ", "a", "\n", "đ")
_VOCAB_C = ("é", "e", "\u0301", "🙂", "\u200d", "👩", "💻", "\u0000")
_VOCAB_D = ("x",)
_VOCAB_E = ("a", "b", "\n", "đ", " ")
_A_ID_TO_CHAR = {0: "b", 1: " ", 2: "a", 3: "\n", 4: "đ"}
_A_CHAR_TO_ID = {"b": 0, " ": 1, "a": 2, "\n": 3, "đ": 4}


def _write_json(path: Path, value: object, *, ensure_ascii: bool = True) -> Path:
    path.write_text(
        json.dumps(value, ensure_ascii=ensure_ascii),
        encoding="utf-8",
    )
    return path


def _tokenizer(
    tmp_path: Path,
    vocabulary: list[str] | tuple[str, ...],
    *,
    name: str = "vocab.json",
) -> CharacterTokenizer:
    path = _write_json(tmp_path / name, list(vocabulary))
    return CharacterTokenizer(str(path))


def _mapping_snapshot(
    tokenizer: CharacterTokenizer,
) -> tuple[dict[int, str], dict[str, int]]:
    return dict(tokenizer.id_to_char), dict(tokenizer.char_to_id)


def test_public_api_signatures_and_positional_keyword_calls(tmp_path: Path):
    assert callable(CharacterTokenizer)
    assert callable(CharacterTokenizer.encode)
    assert callable(CharacterTokenizer.decode)
    assert tuple(inspect.signature(CharacterTokenizer).parameters) == ("vocab_path",)
    assert tuple(inspect.signature(CharacterTokenizer.encode).parameters) == (
        "self",
        "text",
    )
    assert tuple(inspect.signature(CharacterTokenizer.decode).parameters) == (
        "self",
        "token_ids",
    )

    path = _write_json(tmp_path / "api.json", list(_VOCAB_A))
    positional = CharacterTokenizer(str(path))
    keyword = CharacterTokenizer(vocab_path=str(path))

    encoded = positional.encode("b a\nđ")
    assert type(encoded) is list
    assert all(type(token_id) is int for token_id in encoded)
    assert encoded == [0, 1, 2, 3, 4]
    assert keyword.encode(text="aba") == [2, 0, 2]
    assert type(positional.decode([0, 1, 2, 3, 4])) is str
    assert keyword.decode(token_ids=[4, 1, 0]) == "đ b"


def test_vocab_order_and_inverse_mappings_match_file_order(tmp_path: Path):
    tokenizer = _tokenizer(tmp_path, _VOCAB_A)

    assert tokenizer.id_to_char == _A_ID_TO_CHAR
    assert tokenizer.char_to_id == _A_CHAR_TO_ID
    assert tokenizer.encode("b a\nđ") == [0, 1, 2, 3, 4]
    assert tokenizer.decode([0, 1, 2, 3, 4]) == "b a\nđ"


def test_alternate_vocab_order_assigns_different_literal_ids(tmp_path: Path):
    tokenizer = _tokenizer(tmp_path, _VOCAB_E)

    assert tokenizer.encode("aba") == [0, 1, 0]
    assert tokenizer.decode([0, 1, 0]) == "aba"
    assert tokenizer.id_to_char == {0: "a", 1: "b", 2: "\n", 3: "đ", 4: " "}


def test_singleton_vocab_uses_id_zero_for_every_occurrence(tmp_path: Path):
    tokenizer = _tokenizer(tmp_path, _VOCAB_D)

    assert tokenizer.id_to_char == {0: "x"}
    assert tokenizer.char_to_id == {"x": 0}
    assert tokenizer.encode("xxx") == [0, 0, 0]
    assert tokenizer.decode([0, 0, 0]) == "xxx"


def test_literal_and_escaped_json_vocabularies_are_identical(tmp_path: Path):
    literal_path = _write_json(
        tmp_path / "literal.json",
        list(_VOCAB_A),
        ensure_ascii=False,
    )
    escaped_path = _write_json(
        tmp_path / "escaped.json",
        list(_VOCAB_A),
        ensure_ascii=True,
    )
    literal = CharacterTokenizer(str(literal_path))
    escaped = CharacterTokenizer(str(escaped_path))

    for tokenizer in (literal, escaped):
        assert tokenizer.id_to_char == _A_ID_TO_CHAR
        assert tokenizer.char_to_id == _A_CHAR_TO_ID
        assert tokenizer.encode("b a\nđ") == [0, 1, 2, 3, 4]
        assert tokenizer.decode([0, 1, 2, 3, 4]) == "b a\nđ"


@pytest.mark.parametrize(
    ("text", "expected_ids"),
    [
        ("", []),
        ("b", [0]),
        ("đ", [4]),
        ("aba", [2, 0, 2]),
        ("bba", [0, 0, 2]),
        (" ab ", [1, 2, 0, 1]),
        ("\n\n", [3, 3]),
        (" aba\nđ ", [1, 2, 0, 2, 3, 4, 1]),
        ("đ a\nb", [4, 1, 2, 3, 0]),
    ],
)
def test_encode_returns_exact_ids_in_codepoint_order(
    tmp_path: Path,
    text: str,
    expected_ids: list[int],
):
    tokenizer = _tokenizer(tmp_path, _VOCAB_A)

    actual = tokenizer.encode(text)

    assert type(actual) is list
    assert all(type(token_id) is int for token_id in actual)
    assert actual == expected_ids
    assert len(actual) == len(text)


@pytest.mark.parametrize(
    ("token_ids", "expected_text"),
    [
        ([], ""),
        ([0], "b"),
        ([4], "đ"),
        ([4, 1, 0], "đ b"),
        ([0, 0, 2], "bba"),
        ([1, 2, 0, 1], " ab "),
        ([3, 3], "\n\n"),
        ([4, 1, 2, 3, 0], "đ a\nb"),
    ],
)
def test_decode_returns_exact_text_without_mutating_ids(
    tmp_path: Path,
    token_ids: list[int],
    expected_text: str,
):
    tokenizer = _tokenizer(tmp_path, _VOCAB_A)
    original_ids = token_ids.copy()

    actual = tokenizer.decode(token_ids)

    assert type(actual) is str
    assert actual == expected_text
    assert len(actual) == len(token_ids)
    assert token_ids == original_ids


def test_vocab_b_preserves_case_punctuation_and_distinct_whitespace(tmp_path: Path):
    tokenizer = _tokenizer(
        tmp_path,
        ("A", "a", "Đ", "đ", "\t", "\r", "\n", " ", "!", "0"),
    )

    text = "AaĐđ\t\r\n !0"
    expected_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert tokenizer.encode(text) == expected_ids
    assert tokenizer.decode(expected_ids) == text
    assert tokenizer.encode(" \tA\r\n ") == [7, 4, 0, 5, 6, 7]


def test_unicode_is_encoded_as_python_codepoints_without_normalization(tmp_path: Path):
    tokenizer = _tokenizer(tmp_path, _VOCAB_C)

    assert tokenizer.encode("é") == [0]
    assert tokenizer.decode([0]) == "é"
    assert tokenizer.encode("e\u0301") == [1, 2]
    assert tokenizer.decode([1, 2]) == "e\u0301"
    assert tokenizer.encode("🙂") == [3]
    assert tokenizer.decode([3]) == "🙂"
    assert tokenizer.encode("👩\u200d💻") == [5, 4, 6]
    assert tokenizer.decode([5, 4, 6]) == "👩\u200d💻"
    assert tokenizer.encode("\u0000") == [7]
    assert tokenizer.decode([7]) == "\u0000"


def test_unicode_spellings_are_not_interchangeable(tmp_path: Path):
    precomposed_only = _tokenizer(tmp_path, ("é",), name="precomposed.json")
    decomposed_only = _tokenizer(
        tmp_path,
        ("e", "\u0301"),
        name="decomposed.json",
    )

    with pytest.raises(ValueError):
        precomposed_only.encode("e\u0301")
    with pytest.raises(ValueError):
        decomposed_only.encode("é")


def test_multicodepoint_grapheme_is_not_a_single_vocab_token(tmp_path: Path):
    path = _write_json(tmp_path / "grapheme.json", ["👩\u200d💻"])

    with pytest.raises(ValueError):
        CharacterTokenizer(str(path))


@pytest.mark.parametrize(
    ("vocabulary", "text", "token_ids"),
    [
        (_VOCAB_A, "", []),
        (_VOCAB_A, "b a\nđ", [0, 1, 2, 3, 4]),
        (_VOCAB_A, "baab\n", [0, 2, 2, 0, 3]),
        (_VOCAB_D, "", []),
        (_VOCAB_D, "xxx", [0, 0, 0]),
        (_VOCAB_C, "", []),
        (_VOCAB_C, "e\u0301🙂", [1, 2, 3]),
        (_VOCAB_C, "👩\u200d💻", [5, 4, 6]),
    ],
)
def test_encode_decode_round_trips_against_literal_expectations(
    tmp_path: Path,
    vocabulary: tuple[str, ...],
    text: str,
    token_ids: list[int],
):
    tokenizer = _tokenizer(tmp_path, vocabulary)

    assert tokenizer.encode(text) == token_ids
    assert tokenizer.decode(token_ids) == text
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.encode(tokenizer.decode(token_ids)) == token_ids


def test_vocab_a_exhaustively_round_trips_all_id_sequences_through_length_three(
    tmp_path: Path,
):
    tokenizer = _tokenizer(tmp_path, _VOCAB_A)

    for length in range(4):
        for ids in itertools.product(range(5), repeat=length):
            literal_ids = list(ids)
            expected_text = "".join(_VOCAB_A[token_id] for token_id in ids)
            assert tokenizer.decode(literal_ids) == expected_text
            assert tokenizer.encode(expected_text) == literal_ids


def test_long_repeated_text_has_exact_repeated_id_block(tmp_path: Path):
    tokenizer = _tokenizer(tmp_path, _VOCAB_A)

    text = "ab\nđ " * 1_000
    expected_ids = [2, 0, 3, 4, 1] * 1_000

    assert tokenizer.encode(text) == expected_ids
    assert tokenizer.decode(expected_ids) == text


@pytest.mark.parametrize(
    "text",
    [None, False, 1, 1.5, b"aba", ["a", "b"], ("a", "b"), {"a": 1}, iter(("a", "b"))],
)
def test_encode_rejects_non_string_inputs(tmp_path: Path, text: object):
    tokenizer = _tokenizer(tmp_path, _VOCAB_A)

    with pytest.raises(TypeError):
        tokenizer.encode(text)


@pytest.mark.parametrize("text", ["?", "A", "Đ", "\t", "🙂", "?ab", "a?b", "ab?", "ab?đ?"])
def test_encode_rejects_unknown_characters_without_changing_state(
    tmp_path: Path,
    text: str,
):
    tokenizer = _tokenizer(tmp_path, _VOCAB_A)
    original_mappings = _mapping_snapshot(tokenizer)

    with pytest.raises(ValueError):
        tokenizer.encode(text)

    assert _mapping_snapshot(tokenizer) == original_mappings
    assert tokenizer.encode("aba") == [2, 0, 2]
    assert tokenizer.decode([4, 1, 0]) == "đ b"


@pytest.mark.parametrize(
    "token_ids",
    [
        None,
        False,
        1,
        "012",
        b"012",
        (),
        (0, 1),
        {0: "b"},
        range(2),
        iter((0, 1)),
    ],
)
def test_decode_rejects_non_list_containers(tmp_path: Path, token_ids: object):
    tokenizer = _tokenizer(tmp_path, _VOCAB_A)

    with pytest.raises(TypeError):
        tokenizer.decode(token_ids)


@pytest.mark.parametrize("bad_value", [None, 1.0, "1", b"1", [1], {"id": 1}, True, False])
@pytest.mark.parametrize("position", [0, 1, 2])
def test_decode_rejects_non_integer_and_boolean_ids_without_mutation(
    tmp_path: Path,
    bad_value: object,
    position: int,
):
    tokenizer = _tokenizer(tmp_path, _VOCAB_A)
    token_ids: list[object] = [0, 4, 2]
    token_ids[position] = bad_value
    original_ids = token_ids.copy()
    original_mappings = _mapping_snapshot(tokenizer)

    with pytest.raises(TypeError):
        tokenizer.decode(token_ids)

    assert token_ids == original_ids
    assert _mapping_snapshot(tokenizer) == original_mappings
    assert tokenizer.encode("aba") == [2, 0, 2]
    assert tokenizer.decode([4, 1, 0]) == "đ b"


@pytest.mark.parametrize("bad_id", [-1, -5, 5, 6, 10**100, -(10**100)])
@pytest.mark.parametrize("position", [0, 1, 2])
def test_decode_rejects_out_of_range_ids_at_every_position(
    tmp_path: Path,
    bad_id: int,
    position: int,
):
    tokenizer = _tokenizer(tmp_path, _VOCAB_A)
    token_ids = [0, 4, 2]
    token_ids[position] = bad_id
    original_ids = token_ids.copy()
    original_mappings = _mapping_snapshot(tokenizer)

    with pytest.raises(ValueError):
        tokenizer.decode(token_ids)

    assert token_ids == original_ids
    assert _mapping_snapshot(tokenizer) == original_mappings
    assert tokenizer.encode("aba") == [2, 0, 2]
    assert tokenizer.decode([4, 1, 0]) == "đ b"


def test_singleton_vocab_id_zero_is_valid_and_adjacent_ids_are_invalid(tmp_path: Path):
    tokenizer = _tokenizer(tmp_path, _VOCAB_D)

    assert tokenizer.decode([0]) == "x"
    with pytest.raises(ValueError):
        tokenizer.decode([-1])
    with pytest.raises(ValueError):
        tokenizer.decode([1])


@pytest.mark.parametrize(
    "root",
    [None, True, 0, 1.5, "ab", {}, {"a": 0}],
)
def test_constructor_rejects_non_list_json_roots(tmp_path: Path, root: object):
    path = _write_json(tmp_path / "invalid-root.json", root)

    with pytest.raises(TypeError):
        CharacterTokenizer(str(path))


@pytest.mark.parametrize(
    "vocabulary",
    [[None], [True], [0], [1.5], [[]], [{}], ["b", "a", None]],
)
def test_constructor_rejects_non_string_vocabulary_items(
    tmp_path: Path,
    vocabulary: list[object],
):
    path = _write_json(tmp_path / "invalid-item.json", vocabulary)

    with pytest.raises(TypeError):
        CharacterTokenizer(str(path))


@pytest.mark.parametrize(
    "vocabulary",
    [
        [],
        [""],
        ["a", ""],
        ["ab"],
        ["e\u0301"],
        ["👩\u200d💻"],
        ["a", "a"],
        ["b", "a", "b"],
        [" ", " "],
        ["\n", "\n"],
    ],
)
def test_constructor_rejects_empty_duplicate_or_multicodepoint_tokens(
    tmp_path: Path,
    vocabulary: list[str],
):
    path = _write_json(tmp_path / "invalid-vocab.json", vocabulary)

    with pytest.raises(ValueError):
        CharacterTokenizer(str(path))


def test_constructor_rejects_duplicate_from_escaped_json_codepoint(tmp_path: Path):
    path = tmp_path / "escaped-duplicate.json"
    path.write_text(r'["a", "\u0061"]', encoding="utf-8")

    with pytest.raises(ValueError):
        CharacterTokenizer(str(path))


def test_constructor_propagates_missing_path_and_directory_errors(tmp_path: Path):
    missing_path = tmp_path / "missing.json"
    directory_path = tmp_path / "directory"
    directory_path.mkdir()

    with pytest.raises(FileNotFoundError):
        CharacterTokenizer(str(missing_path))
    with pytest.raises(IsADirectoryError):
        CharacterTokenizer(str(directory_path))


@pytest.mark.parametrize(
    "contents",
    ["", "   \n", '["a"', '["a",]', "not json", '["a"] ["b"]'],
)
def test_constructor_propagates_malformed_json_errors(tmp_path: Path, contents: str):
    path = tmp_path / "malformed.json"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        CharacterTokenizer(str(path))


def test_constructor_propagates_invalid_utf8(tmp_path: Path):
    path = tmp_path / "invalid-utf8.json"
    path.write_bytes(b"\xff")

    with pytest.raises(UnicodeDecodeError):
        CharacterTokenizer(str(path))


@pytest.mark.parametrize("error", [PermissionError("denied"), OSError("unavailable")])
def test_constructor_propagates_filesystem_oserrors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: OSError,
):
    path = _write_json(tmp_path / "unreadable.json", list(_VOCAB_A))
    original_open = builtins.open

    def fail_open(file: object, *args: object, **kwargs: object):
        if os.fspath(file) == str(path):
            raise error
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fail_open)

    with pytest.raises(type(error)):
        CharacterTokenizer(str(path))


def test_repeated_successes_and_failures_preserve_mappings_and_outputs(tmp_path: Path):
    tokenizer = _tokenizer(tmp_path, _VOCAB_A)
    original_mappings = _mapping_snapshot(tokenizer)

    assert tokenizer.encode("aba") == [2, 0, 2]
    assert tokenizer.decode([4, 1, 0]) == "đ b"
    with pytest.raises(ValueError):
        tokenizer.encode("a?b")
    with pytest.raises(ValueError):
        tokenizer.decode([0, 5, 4])
    assert tokenizer.decode([0, 0, 2]) == "bba"
    assert tokenizer.encode("b a") == [0, 1, 2]
    assert _mapping_snapshot(tokenizer) == original_mappings


def test_separate_instances_do_not_share_state(tmp_path: Path):
    first_path = _write_json(tmp_path / "first.json", list(_VOCAB_A))
    second_path = _write_json(tmp_path / "second.json", list(_VOCAB_A))
    first = CharacterTokenizer(str(first_path))
    second = CharacterTokenizer(str(second_path))
    first_mappings = _mapping_snapshot(first)
    second_mappings = _mapping_snapshot(second)

    with pytest.raises(ValueError):
        first.encode("?")

    assert first.encode("aba") == [2, 0, 2]
    assert second.encode("aba") == [2, 0, 2]
    assert first.decode([4, 1, 0]) == "đ b"
    assert second.decode([4, 1, 0]) == "đ b"
    assert _mapping_snapshot(first) == first_mappings
    assert _mapping_snapshot(second) == second_mappings


def test_simultaneous_different_vocabularies_keep_independent_ids(tmp_path: Path):
    tokenizer_a = _tokenizer(tmp_path, _VOCAB_A, name="a.json")
    tokenizer_e = _tokenizer(tmp_path, _VOCAB_E, name="e.json")

    assert tokenizer_a.encode("aba") == [2, 0, 2]
    assert tokenizer_e.encode("aba") == [0, 1, 0]
    assert tokenizer_e.decode([0, 1, 0]) == "aba"
    assert tokenizer_a.decode([2, 0, 2]) == "aba"
    assert tokenizer_a.encode("aba") == [2, 0, 2]
    assert tokenizer_e.encode("aba") == [0, 1, 0]


def test_existing_instance_keeps_ids_after_vocab_file_is_replaced(tmp_path: Path):
    path = _write_json(tmp_path / "replace.json", list(_VOCAB_A))
    existing = CharacterTokenizer(str(path))

    _write_json(path, list(_VOCAB_E))
    replacement = CharacterTokenizer(str(path))

    assert existing.encode("aba") == [2, 0, 2]
    assert existing.decode([2, 0, 2]) == "aba"
    assert replacement.encode("aba") == [0, 1, 0]
    assert replacement.decode([0, 1, 0]) == "aba"


def test_existing_instance_keeps_vocab_after_source_becomes_malformed(tmp_path: Path):
    path = _write_json(tmp_path / "malformed-after-load.json", list(_VOCAB_A))
    tokenizer = CharacterTokenizer(str(path))

    path.write_text("{broken", encoding="utf-8")

    assert tokenizer.encode("aba") == [2, 0, 2]
    assert tokenizer.decode([4, 1, 0]) == "đ b"


def test_existing_instance_survives_source_deletion(tmp_path: Path):
    path = _write_json(tmp_path / "deleted.json", list(_VOCAB_A))
    tokenizer = CharacterTokenizer(str(path))

    path.unlink()

    assert tokenizer.encode("aba") == [2, 0, 2]
    assert tokenizer.decode([4, 1, 0]) == "đ b"
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""
    with pytest.raises(FileNotFoundError):
        CharacterTokenizer(str(path))


def test_mutating_encode_result_does_not_change_later_results_or_mappings(tmp_path: Path):
    tokenizer = _tokenizer(tmp_path, _VOCAB_A)
    original_mappings = _mapping_snapshot(tokenizer)
    encoded = tokenizer.encode("aba")
    encoded[0] = 999

    assert tokenizer.encode("aba") == [2, 0, 2]
    assert _mapping_snapshot(tokenizer) == original_mappings


def test_decode_uses_current_caller_list_without_mutating_it(tmp_path: Path):
    tokenizer = _tokenizer(tmp_path, _VOCAB_A)
    token_ids = [4, 1, 0]

    assert tokenizer.decode(token_ids) == "đ b"
    assert token_ids == [4, 1, 0]
    assert tokenizer.decode(token_ids) == "đ b"
    token_ids[0] = 0
    assert tokenizer.decode(token_ids) == "b b"
    assert token_ids == [0, 1, 0]
