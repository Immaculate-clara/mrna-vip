"""Module 2 - scaled dot-product self-attention over an mRNA sequence, from scratch.

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

Satisfies FR-2.1 .. FR-2.5. Pure NumPy (the PRD allows NumPy or PyTorch).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config

VOCAB = ("A", "C", "G", "U")
EMBED_DIM = 8
MIN_LEN, MAX_LEN = 5, 50
DEFAULT_SEQUENCE = "ACGUACGUACGGCUA"


def make_embedding_table(embed_dim: int = EMBED_DIM, seed: int = config.RANDOM_SEED) -> dict[str, np.ndarray]:
    """One random vector per nucleotide (a 'learnable' table, here only initialised)."""
    rng = np.random.default_rng(seed)
    return {base: rng.standard_normal(embed_dim) for base in VOCAB}


EMBEDDING_TABLE = make_embedding_table()


def validate_sequence(sequence: str) -> str:
    """Upper-case and validate: only A/C/G/U, length 5-50 (FR-2.1, FR-2.2)."""
    if not isinstance(sequence, str):
        raise TypeError("sequence must be a string")
    seq = sequence.strip().upper()
    bad = sorted({ch for ch in seq if ch not in VOCAB})
    if bad:
        raise ValueError(f"Invalid nucleotide(s) {bad}: only A, C, G, U are allowed")
    return seq


def positional_encoding(seq_len: int, embed_dim: int = EMBED_DIM) -> np.ndarray:
    """Sinusoidal positional encoding (Vaswani et al., 2017).

    Without it, self-attention is permutation-invariant: two positions holding the
    same base get *identical* attention rows. Adding it makes position matter.
    """
    pos = np.arange(seq_len)[:, None]
    i = np.arange(embed_dim)[None, :]
    angles = pos / np.power(10000.0, (2 * (i // 2)) / embed_dim)
    return np.where(i % 2 == 0, np.sin(angles), np.cos(angles))


def encode_sequence(sequence: str, table: dict[str, np.ndarray] | None = None,
                    add_positions: bool = False) -> np.ndarray:
    """Turn 'ACGU...' into a (seq_len, embed_dim) matrix.

    The 5-50 length rule applies to the project's sequences; very short inputs such
    as 'ACGU' (4 bases) are still encodable because FR-2.1's own example uses one,
    so only the upper bound and the alphabet are hard errors here.
    """
    seq = validate_sequence(sequence)
    if len(seq) == 0:
        raise ValueError("sequence must not be empty")
    if len(seq) > MAX_LEN:
        raise ValueError(f"sequence length {len(seq)} exceeds the maximum of {MAX_LEN}")
    table = table or EMBEDDING_TABLE
    X = np.stack([table[base] for base in seq])
    return X + positional_encoding(len(seq), X.shape[1]) if add_positions else X


def softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax; subtracting the row max avoids overflow in exp()."""
    shifted = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=-1, keepdims=True)


def self_attention(X: np.ndarray, d_k: int = EMBED_DIM, seed: int = config.RANDOM_SEED):
    """Single-head self-attention.

    Returns (attention_weights (n, n), output (n, d_k)). Projection matrices are
    random (scaled by 0.1) and seeded, so results are reproducible.
    """
    n, embed_dim = X.shape
    rng = np.random.default_rng(seed)
    W_q, W_k, W_v = (rng.standard_normal((embed_dim, d_k)) * 0.1 for _ in range(3))
    Q, K, V = X @ W_q, X @ W_k, X @ W_v
    scores = (Q @ K.T) / np.sqrt(d_k)
    weights = softmax(scores)
    return weights, weights @ V


def multi_head_attention(X: np.ndarray, n_heads: int = 4, d_k: int = 4, seed: int = config.RANDOM_SEED):
    """Bonus: independent Q/K/V per head; head outputs are concatenated.

    Returns (list of per-head weights, concatenated output (n, n_heads * d_k)).
    """
    heads = [self_attention(X, d_k=d_k, seed=seed + h) for h in range(n_heads)]
    return [w for w, _ in heads], np.concatenate([o for _, o in heads], axis=-1)


def plot_attention(weights: np.ndarray, sequence: str, save_path: str | Path | None, ax=None) -> None:
    """Labelled heatmap; both axes show position index and base (FR-2.5)."""
    labels = [f"{i + 1}:{b}" for i, b in enumerate(sequence)]
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(weights, cmap="viridis")
    ax.set_xticks(range(len(sequence)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(sequence)), labels, fontsize=7)
    ax.set_xlabel("Attending TO position (index:base)")
    ax.set_ylabel("Attending FROM position (index:base)")
    ax.set_title("mRNA Sequence Self-Attention Weights")
    ax.figure.colorbar(im, ax=ax, label="Attention weight", fraction=0.046)
    if own_fig:
        fig.tight_layout()
        if save_path is not None:  # with no path the figure is left open for plt.show()
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
            plt.close(fig)


def run(sequence: str = DEFAULT_SEQUENCE, output_dir: Path = config.OUTPUT_DIR) -> np.ndarray:
    output_dir = Path(output_dir)
    seq = validate_sequence(sequence)
    X = encode_sequence(seq)
    weights, _ = self_attention(X)
    assert np.allclose(weights.sum(axis=1), 1.0), "softmax rows must sum to 1"  # FR-2.4
    plot_attention(weights, seq, output_dir / "figures" / "attention_heatmap.png")
    np.savez(output_dir / "attention_result.npz", weights=weights, sequence=np.array(seq))
    top = np.unravel_index(np.argmax(weights - np.eye(len(seq)) * 1e9), weights.shape)
    print(f"{seq}: strongest cross-position attention {top[0] + 1}:{seq[top[0]]} -> "
          f"{top[1] + 1}:{seq[top[1]]} ({weights[top]:.3f})")
    return weights


if __name__ == "__main__":
    run()
