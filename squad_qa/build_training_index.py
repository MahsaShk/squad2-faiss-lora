"""Build the shared train/tuning FAISS index without loading a QA model or chat tokenizer."""
import argparse
import os

import faiss
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from squad_qa.io import load_json, save_json
from squad_qa.prepare_lora_data import (
    DATASET, MANIFEST_PATH, TRAIN_PATH, TUNING_PATH, check_title_overlap,
)
from squad_qa.retrieval import (
    EMBED_MODEL, TRAIN_INDEX_PATH as INDEX_PATH, TRAIN_OWNERS_PATH as OWNERS_PATH,
    TRAIN_PARAGRAPHS_PATH as PARAGRAPHS_PATH, chunk_text, encode,
)


def collect_paragraphs(squad, indexed_titles):
    paragraphs, pid_of = [], {}
    for title, context in zip(squad["title"], squad["context"]):
        if title in indexed_titles and context not in pid_of:
            pid_of[context] = len(paragraphs)
            paragraphs.append({"title": title, "context": context})
    return paragraphs, pid_of


def build_index(model, paragraphs):
    tok = model.tokenizer
    window = model.max_seq_length - 2
    texts, owners = [], []
    for pid, para in enumerate(paragraphs):
        pieces = chunk_text(tok, para["context"], window)
        texts.extend(pieces)
        owners.extend([pid] * len(pieces))
    print(f"Index: {len(texts)} chunks from {len(paragraphs)} paragraphs")
    index = faiss.IndexFlatIP(model.get_sentence_embedding_dimension())
    index.add(encode(model, texts))
    os.makedirs(os.path.dirname(INDEX_PATH) or ".", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    save_json(OWNERS_PATH, owners)
    save_json(PARAGRAPHS_PATH, paragraphs)
    return index, owners


def main(device):
    manifest = load_json(MANIFEST_PATH)
    train_qs, tuning_qs = load_json(TRAIN_PATH), load_json(TUNING_PATH)
    title_sets = {name: set(manifest["titles"][name])
                  for name in ("train", "tuning", "validation")}
    check_title_overlap(title_sets)
    for name, records in (("train", train_qs), ("tuning", tuning_qs)):
        if any(q["title"] not in title_sets[name] for q in records):
            raise ValueError(f"Question outside assigned {name} titles")
    squad = load_dataset(DATASET, split="train")
    paragraphs, pid_of = collect_paragraphs(squad, title_sets["train"] | title_sets["tuning"])
    if any(q["context"] not in pid_of for q in train_qs + tuning_qs):
        raise ValueError("Gold paragraph missing from training/tuning index")
    print(f"Embedding device: {device}")
    model = SentenceTransformer(EMBED_MODEL, device=device)
    build_index(model, paragraphs)
    print(f"Wrote {INDEX_PATH}, {OWNERS_PATH}, and {PARAGRAPHS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"],
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    main(args.device)
