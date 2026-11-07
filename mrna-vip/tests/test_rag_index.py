import json

from src import config
from src.rag_index import GuidelineIndex, chunk_text, count_tokens, query_guidelines


def _chunks():
    return chunk_text(config.GUIDELINES_TXT.read_text(encoding="utf-8"))


def test_chunk_sizes_within_window():  # FR-3.1
    for c in _chunks():
        assert config.CHUNK_MIN_TOKENS <= count_tokens(c) <= config.CHUNK_TARGET_MAX_TOKENS


def test_consecutive_chunks_overlap():
    chunks = _chunks()
    for a, b in zip(chunks, chunks[1:]):
        first_line = b.split("\n")[0][:40]
        assert first_line in a


def test_long_text_gets_many_overlapping_chunks():
    text = "\n\n".join(f"Sentence number {i} says something. Another one follows here." for i in range(300))
    chunks = chunk_text(text)
    assert len(chunks) > 3
    assert all(count_tokens(c) <= 500 for c in chunks)


def test_embeddings_equal_chunks_and_topk(index):  # FR-3.2, FR-3.4
    assert index.store.count() == len(index.chunks)
    for q in ["storage temperature", "dose for elderly", "adverse event reporting",
              "allergy contraindication", "monitoring after injection"]:
        chunks, latency = query_guidelines(index, q, k=2)
        assert len(chunks) == 2 and 0 < latency < 5  # FR-3.5


def test_scores_descending(index):
    res, _ = index.query("secondary clinician review", k=2)
    assert res[0].score >= res[1].score


def test_persistence_survives_reload(index):  # FR-3.3
    reloaded = GuidelineIndex.load(index.persist_dir, log_path=None)
    a, _ = index.query("how long to monitor patients", k=2)
    b, _ = reloaded.query("how long to monitor patients", k=2)
    assert [r.chunk_id for r in a] == [r.chunk_id for r in b]


def test_latency_logged(index):
    index.query("cold chain transport", k=1)
    last = json.loads(index.log_path.read_text().splitlines()[-1])
    assert last["latency_s"] > 0 and last["question"] == "cold chain transport"
