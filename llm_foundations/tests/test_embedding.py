"""Kiểm tra hợp đồng tra cứu embedding theo token ID."""

import pytest
import torch

from llm_foundations.embedding import embedding_lookup


_DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    _DEVICES.append(torch.device("cuda"))

_CANONICAL_WEIGHT_VALUES = [
    [0.0, 0.1, 0.2],
    [1.0, 1.1, 1.2],
    [2.0, 2.1, 2.2],
    [3.0, 3.1, 3.2],
    [4.0, 4.1, 4.2],
]


def _canonical_weight(
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    return torch.tensor(_CANONICAL_WEIGHT_VALUES, dtype=dtype, device=device)


def test_repeated_ids_preserve_order_and_occurrences():
    token_ids = torch.tensor([[1, 1, 4, 0]], dtype=torch.long)
    weight = _canonical_weight()
    expected = torch.tensor(
        [[[1.0, 1.1, 1.2], [1.0, 1.1, 1.2], [4.0, 4.1, 4.2], [0.0, 0.1, 0.2]]],
        dtype=torch.float64,
    )

    result = embedding_lookup(token_ids, weight)

    assert result.shape == (1, 4, 3)
    assert torch.equal(result, expected)


def test_batch_positions_share_one_weight_table():
    token_ids = torch.tensor([[2, 0], [1, 2]], dtype=torch.long)
    weight = _canonical_weight()
    expected = torch.tensor(
        [
            [[2.0, 2.1, 2.2], [0.0, 0.1, 0.2]],
            [[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]],
        ],
        dtype=torch.float64,
    )
    expected_row_2 = torch.tensor([2.0, 2.1, 2.2], dtype=torch.float64)

    result = embedding_lookup(token_ids, weight)

    assert result.shape == (2, 2, 3)
    assert torch.equal(result, expected)
    assert torch.equal(result[0, 0], expected_row_2)
    assert torch.equal(result[1, 1], expected_row_2)


def test_rectangular_token_grid_preserves_all_axes():
    token_ids = torch.tensor([[4, 3, 2, 1], [0, 2, 4, 1]], dtype=torch.long)
    weight = _canonical_weight()
    expected = torch.tensor(
        [
            [
                [4.0, 4.1, 4.2],
                [3.0, 3.1, 3.2],
                [2.0, 2.1, 2.2],
                [1.0, 1.1, 1.2],
            ],
            [
                [0.0, 0.1, 0.2],
                [2.0, 2.1, 2.2],
                [4.0, 4.1, 4.2],
                [1.0, 1.1, 1.2],
            ],
        ],
        dtype=torch.float64,
    )

    result = embedding_lookup(token_ids, weight)

    assert result.shape == (2, 4, 3)
    assert torch.equal(result, expected)


def test_boundary_ids_do_not_mutate_inputs():
    token_ids = torch.tensor([[0, 4], [4, 0]], dtype=torch.long)
    weight = _canonical_weight()
    original_token_ids = token_ids.clone()
    original_weight = weight.clone()
    expected = torch.tensor(
        [
            [[0.0, 0.1, 0.2], [4.0, 4.1, 4.2]],
            [[4.0, 4.1, 4.2], [0.0, 0.1, 0.2]],
        ],
        dtype=torch.float64,
    )

    result = embedding_lookup(token_ids, weight)

    assert result.shape == (2, 2, 3)
    assert torch.equal(result, expected)
    assert torch.equal(token_ids, original_token_ids)
    assert torch.equal(weight, original_weight)
    assert result.dtype == weight.dtype
    assert result.device == weight.device


@pytest.mark.parametrize(
    "ids, weight_values, expected_values, expected_shape",
    [
        pytest.param(
            [[3, 0]],
            None,
            [[[3.0, 3.1, 3.2], [0.0, 0.1, 0.2]]],
            (1, 2, 3),
            id="B=1",
        ),
        pytest.param(
            [[3], [0]],
            None,
            [[[3.0, 3.1, 3.2]], [[0.0, 0.1, 0.2]]],
            (2, 1, 3),
            id="T=1",
        ),
        pytest.param(
            [[0, 0], [0, 0]],
            [[7.0, 8.0, 9.0]],
            [
                [[7.0, 8.0, 9.0], [7.0, 8.0, 9.0]],
                [[7.0, 8.0, 9.0], [7.0, 8.0, 9.0]],
            ],
            (2, 2, 3),
            id="V=1",
        ),
        pytest.param(
            [[2, 0], [1, 2]],
            [[10.0], [20.0], [30.0]],
            [[[30.0], [10.0]], [[20.0], [30.0]]],
            (2, 2, 1),
            id="D=1",
        ),
    ],
)
def test_singleton_dimensions_are_preserved(ids, weight_values, expected_values, expected_shape):
    token_ids = torch.tensor(ids, dtype=torch.long)
    weight = (
        _canonical_weight()
        if weight_values is None
        else torch.tensor(weight_values, dtype=torch.float64)
    )
    expected = torch.tensor(expected_values, dtype=torch.float64)

    result = embedding_lookup(token_ids, weight)

    assert result.shape == expected_shape
    assert torch.equal(result, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", _DEVICES)
def test_preserves_weight_dtype_and_device(dtype: torch.dtype, device: torch.device):
    token_ids = torch.tensor([[4, 1]], dtype=torch.long, device=device)
    weight = _canonical_weight(dtype=dtype, device=device)
    expected = torch.tensor(
        [[[4.0, 4.1, 4.2], [1.0, 1.1, 1.2]]],
        dtype=dtype,
        device=device,
    )

    result = embedding_lookup(token_ids, weight)

    assert result.shape == (1, 2, 3)
    assert torch.equal(result, expected)
    assert result.dtype == weight.dtype
    assert result.device == weight.device
