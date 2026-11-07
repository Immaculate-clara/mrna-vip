"""Tiny text helpers kept in their own module so pickled objects that reference them
unpickle correctly no matter which script built them (``python -m src.rag_index`` would
otherwise record ``__main__`` as the function's module)."""
from __future__ import annotations

import re


def stem_tokenize(text: str) -> list[str]:
    """Lower-case words with a crude suffix stripper so 'monitored' ~ 'monitor'.

    Module-level (not a lambda) so a fitted vectorizer can be pickled by joblib.
    """
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        for suffix in ("ation", "ing", "ed", "es", "ly", "s"):
            if len(w) > len(suffix) + 3 and w.endswith(suffix):
                w = w[: -len(suffix)]
                break
        out.append(w)
    return out
