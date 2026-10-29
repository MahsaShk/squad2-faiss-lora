"""Shared passage chunking, FAISS search, and question-to-context retrieval."""
import gc
import os

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from squad_qa.io import load_json
from squad_qa.prepare_lora_data import MANIFEST_PATH

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = os.path.join("data", "eval.faiss")
PASSAGES_PATH = os.path.join("data", "eval_passages.json")
CHUNKED_INDEX_PATH = os.path.join("data", "eval_chunked.faiss")
CHUNKED_OWNERS_PATH = os.path.join("data", "eval_chunked_owners.json")
TRAIN_INDEX_PATH = os.path.join("data", "lora_index.faiss")
TRAIN_OWNERS_PATH = os.path.join("data", "lora_index_owners.json")
TRAIN_PARAGRAPHS_PATH = os.path.join("data", "lora_index_paragraphs.json")
STRIDE = 127
TOP_K = 3
SEARCH_HITS = 50


def chunk_text(tok, text, window):
    """Split text into overlapping token windows; short text is returned unchanged."""
    offs = tok(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    if len(offs) <= window:
        return [text]
    pieces, start = [], 0
    while True:
        end = min(start + window, len(offs))
        pieces.append(text[offs[start][0]:offs[end - 1][1]])
        if end == len(offs):
            return pieces
        start += STRIDE


def encode(model, texts):
    vectors = model.encode(
        texts, batch_size=64, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=True,
    )
    return np.asarray(vectors, dtype="float32")


def dedupe_owners(row, owners):
    """Map chunk hits to distinct parent IDs, preserving rank and ignoring FAISS padding."""
    return list(dict.fromkeys(owners[int(hit)] for hit in row if hit >= 0))


def search(model, index, owners, questions):
    if index.ntotal == 0:
        raise ValueError("Cannot search an empty FAISS index")
    if len(owners) != index.ntotal:
        raise ValueError("FAISS owners do not match index")
    vectors = model.encode(
        [q["question"] for q in questions], normalize_embeddings=True, convert_to_numpy=True
    )
    _, hits = index.search(np.asarray(vectors, dtype="float32"), min(SEARCH_HITS, index.ntotal))
    return [dedupe_owners(row, owners) for row in hits]


def retrieval_contexts(records, split, device):
    if split not in ("tuning", "dev"):
        raise ValueError("Retrieval split must be tuning or dev")
    if split == "tuning":
        index_path, owners_path = TRAIN_INDEX_PATH, TRAIN_OWNERS_PATH
        paragraphs = load_json(TRAIN_PARAGRAPHS_PATH)
        manifest = load_json(MANIFEST_PATH)
        allowed = set(manifest["titles"]["train"]) | set(manifest["titles"]["tuning"])
        if any(p["title"] not in allowed for p in paragraphs):
            raise ValueError("Train/tuning index contains titles outside the split manifest")
        passages = [p["context"] for p in paragraphs]
    else:
        index_path, owners_path = CHUNKED_INDEX_PATH, CHUNKED_OWNERS_PATH
        passages = load_json(PASSAGES_PATH)
    index = faiss.read_index(index_path)
    owners = load_json(owners_path)
    if len(owners) != index.ntotal or any(not 0 <= p < len(passages) for p in owners):
        raise ValueError("FAISS owners do not match index/passages")
    embedder = SentenceTransformer(EMBED_MODEL, device=device)
    ranked_ids = search(embedder, index, owners, records)
    del embedder
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    contexts, ranked = [], []
    for distinct in ranked_ids:
        if len(distinct) < TOP_K:
            raise ValueError("Retrieval returned fewer than three distinct paragraphs")
        ranked.append([passages[pid] for pid in distinct])
        contexts.append(ranked[-1][:TOP_K])
    recall = {}
    for k in (1, 3, 5):
        for name, selected in (
            ("all", list(range(len(records)))),
            ("hasans", [i for i, r in enumerate(records) if r["answers"]["text"]]),
        ):
            recall[f"recall@{k}_{name}"] = (
                sum(records[i].get("context") in ranked[i][:k] for i in selected) / len(selected)
                if selected else None
            )
    return contexts, recall
