import numpy as np
import pytest

from src.attention import (encode_sequence, multi_head_attention, plot_attention, run,
                           self_attention, softmax)


def test_encode_shape():  # FR-2.1
    assert encode_sequence("ACGU").shape == (4, 8)


@pytest.mark.parametrize("bad", ["ACGX", "ACGT", "acg1u"])
def test_invalid_characters_rejected(bad):  # FR-2.2
    with pytest.raises(ValueError):
        encode_sequence(bad)


def test_too_long_rejected():
    with pytest.raises(ValueError):
        encode_sequence("A" * 51)


def test_weights_shape_and_rows_sum_to_one():  # FR-2.3, FR-2.4
    for seq in ["ACGUACGUACGGCUA", "GGGGGCCCCCAAAAAUUUUU", "ACGUA"]:
        w, out = self_attention(encode_sequence(seq))
        assert w.shape == (len(seq), len(seq)) and np.allclose(w.sum(axis=1), 1.0)
        assert (w >= 0).all() and out.shape == (len(seq), 8)


def test_softmax_is_stable_for_large_values():
    assert np.allclose(softmax(np.array([[1000.0, 1001.0]])).sum(), 1.0)


def test_deterministic_and_positional_encoding_breaks_symmetry():
    X = encode_sequence("AAAAAA")
    w, _ = self_attention(X)
    assert np.allclose(w, w[0])  # identical tokens -> identical rows
    w2, _ = self_attention(encode_sequence("AAAAAA", add_positions=True))
    assert not np.allclose(w2, w2[0])


def test_multi_head():
    heads, out = multi_head_attention(encode_sequence("ACGUACGU"), n_heads=3, d_k=4)
    assert len(heads) == 3 and out.shape == (8, 12)


def test_heatmap_saved(tmp_path):  # FR-2.5
    run(output_dir=tmp_path)
    assert (tmp_path / "figures" / "attention_heatmap.png").stat().st_size > 1000
