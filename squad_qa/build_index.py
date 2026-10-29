"""Build the FAISS eval index over the unique validation-split paragraphs.

Variants:
  flat    - one vector per paragraph, truncated at the embedder's token limit (eval.faiss).
  chunked - long paragraphs split into overlapping token windows; each chunk vector
            maps back to its parent paragraph (eval_chunked.faiss + owners file).
"""
import argparse
import json
import os

import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from squad_qa.retrieval import (
    CHUNKED_INDEX_PATH, CHUNKED_OWNERS_PATH, EMBED_MODEL, INDEX_PATH, PASSAGES_PATH,
    STRIDE, chunk_text, encode,
)


def build_flat(model, passages):
    too_long = sum(
        len(model.tokenizer.encode(p, add_special_tokens=True)) > model.max_seq_length
        for p in passages
    )
    print(f"Passages: {len(passages)}; longer than max_seq_length={model.max_seq_length}: {too_long}")
    index = faiss.IndexFlatIP(model.get_sentence_embedding_dimension())
    index.add(encode(model, passages))
    faiss.write_index(index, INDEX_PATH)
    with open(PASSAGES_PATH, "w", encoding="utf-8") as f:
        json.dump(passages, f, ensure_ascii=False)
    print(f"Indexed {index.ntotal} passages to {INDEX_PATH}")


def build_chunked(model, passages):
    tok = model.tokenizer
    window = model.max_seq_length - 2
    texts, owners = [], []
    for pid, p in enumerate(passages):
        pieces = chunk_text(tok, p, window)
        texts.extend(pieces)
        owners.extend([pid] * len(pieces))
    print(f"Chunks: {len(texts)} from {len(passages)} paragraphs (window={window} tokens, stride={STRIDE})")
    index = faiss.IndexFlatIP(model.get_sentence_embedding_dimension())
    index.add(encode(model, texts))
    faiss.write_index(index, CHUNKED_INDEX_PATH)
    with open(CHUNKED_OWNERS_PATH, "w", encoding="utf-8") as f:
        json.dump(owners, f)
    print(f"Indexed {index.ntotal} chunks to {CHUNKED_INDEX_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["flat", "chunked"], default="chunked")
    args = parser.parse_args()
    os.makedirs("data", exist_ok=True)
    model = SentenceTransformer(EMBED_MODEL, device="cuda")
    if args.variant == "flat":
        val = load_dataset("rajpurkar/squad_v2", split="validation")
        build_flat(model, sorted(set(val["context"])))
    else:
        with open(PASSAGES_PATH, encoding="utf-8") as f:
            passages = json.load(f)
        build_chunked(model, passages)


if __name__ == "__main__":
    main()
