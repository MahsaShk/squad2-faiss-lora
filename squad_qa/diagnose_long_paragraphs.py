"""Retrieval-only diagnostic for paragraphs longer than the embedder's token limit.

Compares the current eval index (one vector per paragraph, truncated at the limit)
with an index where long paragraphs are split into overlapping token windows.
Only questions whose gold paragraph is long are scored. No LLM is run.
"""
import json

import faiss
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from squad_qa.retrieval import EMBED_MODEL, INDEX_PATH, PASSAGES_PATH, chunk_text

KS = (1, 3, 5)
MAX_CHUNK_HITS = 50


def truncation_char(tok, text, window):
    offs = tok(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    return offs[window - 1][1] if len(offs) > window else len(text)


def recall(ranked, gold, mask, k):
    hits = [g in r[:k] for r, g, m in zip(ranked, gold, mask) if m]
    return 100.0 * sum(hits) / len(hits)


def main():
    model = SentenceTransformer(EMBED_MODEL, device="cuda")
    tok = model.tokenizer
    window = model.max_seq_length - 2

    with open(PASSAGES_PATH, encoding="utf-8") as f:
        passages = json.load(f)
    pid_of = {p: i for i, p in enumerate(passages)}
    eval_index = faiss.read_index(INDEX_PATH)
    assert eval_index.ntotal == len(passages)

    long_ids = {
        i for i, p in enumerate(passages)
        if len(tok.encode(p, add_special_tokens=True)) > model.max_seq_length
    }
    print(f"long paragraphs: {len(long_ids)} of {len(passages)}")

    chunk_texts, owners = [], []
    for pid, p in enumerate(passages):
        pieces = chunk_text(tok, p, window) if pid in long_ids else [p]
        chunk_texts.extend(pieces)
        owners.extend([pid] * len(pieces))
    print(f"chunks: {len(chunk_texts)} (from {len(passages)} paragraphs)")

    chunk_vecs = model.encode(
        chunk_texts, batch_size=64, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    )
    chunk_index = faiss.IndexFlatIP(chunk_vecs.shape[1])
    chunk_index.add(np.asarray(chunk_vecs, dtype="float32"))

    val = load_dataset("rajpurkar/squad_v2", split="validation")
    rows = [r for r in val if pid_of[r["context"]] in long_ids]
    gold = [pid_of[r["context"]] for r in rows]
    has_ans = [len(r["answers"]["text"]) > 0 for r in rows]
    q_vecs = np.asarray(
        model.encode(
            [r["question"] for r in rows], batch_size=128,
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
        ),
        dtype="float32",
    )

    _, cur_ids = eval_index.search(q_vecs, max(KS))
    cur_ranked = cur_ids.tolist()
    _, ch_ids = chunk_index.search(q_vecs, MAX_CHUNK_HITS)
    ch_ranked = []
    for row in ch_ids:
        seen = []
        for c in row:
            pid = owners[c]
            if pid not in seen:
                seen.append(pid)
        ch_ranked.append(seen)

    n_ans = sum(has_ans)
    print(f"questions on long paragraphs: {len(rows)} (answerable {n_ans}, unanswerable {len(rows) - n_ans})")
    groups = {"all": [True] * len(rows), "answerable": has_ans}
    print(f"{'set':<12}{'k':>3}{'current %':>12}{'chunked %':>12}")
    for name, mask in groups.items():
        for k in KS:
            print(f"{name:<12}{k:>3}{recall(cur_ranked, gold, mask, k):>12.1f}{recall(ch_ranked, gold, mask, k):>12.1f}")

    boundary_cache = {}
    past_cut, straddles = 0, 0
    for r in rows:
        if not r["answers"]["text"]:
            continue
        ctx = r["context"]
        if ctx not in boundary_cache:
            boundary_cache[ctx] = truncation_char(tok, ctx, window)
        b = boundary_cache[ctx]
        start = r["answers"]["answer_start"][0]
        end = start + len(r["answers"]["text"][0])
        if start >= b:
            past_cut += 1
        elif end > b:
            straddles += 1
    print(f"answerable answers entirely past the truncation point: {past_cut} of {n_ans}")
    print(f"answerable answers cut by the truncation point: {straddles} of {n_ans}")


if __name__ == "__main__":
    main()
