"""Prepare balanced, title-disjoint training and tuning subsets for LoRA."""
import argparse
import copy
import hashlib
import json
import os
import random
from itertools import combinations

from datasets import load_dataset

from squad_qa.data_prep import OUT_PATH as DEV_PATH, SEED

DATASET = "rajpurkar/squad_v2"
TRAIN_SIZE = 3000
TUNING_SIZE = 300
TUNING_TITLE_COUNT = 30
TRAIN_PATH = os.path.join("data", "train_3000.json")
TUNING_PATH = os.path.join("data", "tuning_300.json")
MANIFEST_PATH = os.path.join("data", "lora_split_manifest.json")


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def check_title_overlap(title_sets):
    for left, right in combinations(title_sets, 2):
        overlap = title_sets[left] & title_sets[right]
        if overlap:
            raise ValueError(f"Title leakage between {left} and {right}: {sorted(overlap)}")


def sample_balanced(records, size, rng):
    if size <= 0 or size % 2:
        raise ValueError("Subset size must be a positive even number.")
    has_ans = [record for record in records if record["answers"]["text"]]
    no_ans = [record for record in records if not record["answers"]["text"]]
    per_group = size // 2
    if min(len(has_ans), len(no_ans)) < per_group:
        raise ValueError(
            f"Need {per_group} questions per answerability group; "
            f"found {len(has_ans)} answerable and {len(no_ans)} unanswerable."
        )
    selected = rng.sample(has_ans, per_group) + rng.sample(no_ans, per_group)
    rng.shuffle(selected)
    return [
        {
            "id": record["id"],
            "title": record["title"],
            "context": record["context"],
            "question": record["question"],
            "answers": {
                "text": list(record["answers"]["text"]),
                "answer_start": list(record["answers"]["answer_start"]),
            },
        }
        for record in selected
    ]


def make_splits(
    records,
    validation_titles,
    seed=SEED,
    tuning_title_count=TUNING_TITLE_COUNT,
    train_size=TRAIN_SIZE,
    tuning_size=TUNING_SIZE,
):
    titles = sorted({record["title"] for record in records})
    if not 0 < tuning_title_count < len(titles):
        raise ValueError("Tuning title count must leave titles in both splits.")
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Source training data contains duplicate question IDs.")

    rng = random.Random(seed)
    tuning_titles = set(rng.sample(titles, tuning_title_count))
    train_titles = set(titles) - tuning_titles
    title_sets = {
        "train": train_titles,
        "tuning": tuning_titles,
        "validation": set(validation_titles),
    }
    check_title_overlap(title_sets)

    train = sample_balanced(
        [record for record in records if record["title"] in train_titles],
        train_size,
        rng,
    )
    tuning = sample_balanced(
        [record for record in records if record["title"] in tuning_titles],
        tuning_size,
        rng,
    )
    manifest = {
        "dataset": DATASET,
        "seed": seed,
        "source_split": "train",
        "evaluation_source_split": "validation",
        "titles": {name: sorted(group) for name, group in title_sets.items()},
        "counts": {
            name: {
                "questions": len(subset),
                "answerable": sum(bool(record["answers"]["text"]) for record in subset),
                "unanswerable": sum(not record["answers"]["text"] for record in subset),
            }
            for name, subset in (("train", train), ("tuning", tuning))
        },
    }
    return train, tuning, manifest


def expand_training(records, existing_train, tuning, dev, manifest, size, seed=SEED):
    if size <= len(existing_train) or size % 2:
        raise ValueError("Expanded size must be even and larger than the existing training set.")
    title_sets = {name: set(manifest["titles"][name])
                  for name in ("train", "tuning", "validation")}
    check_title_overlap(title_sets)
    source = {r["id"]: r for r in records}
    if len(source) != len(records):
        raise ValueError("Source training data contains duplicate question IDs.")
    seen = set()
    for name, subset in (("train", existing_train), ("tuning", tuning), ("validation", dev)):
        for record in subset:
            if record["id"] in seen:
                raise ValueError("Existing splits contain duplicate or overlapping question IDs.")
            seen.add(record["id"])
            if record["title"] not in title_sets[name]:
                raise ValueError(f"Question outside assigned {name} titles: {record['id']}")
            if name != "validation" and source.get(record["id"]) != record:
                raise ValueError(f"Existing question differs from source training data: {record['id']}")
            if name == "validation" and record["id"] in source:
                raise ValueError("Existing dev set shares question IDs with training data.")
    if sum(bool(r["answers"]["text"]) for r in existing_train) * 2 != len(existing_train):
        raise ValueError("Existing training set must be balanced before expansion.")
    rng = random.Random(seed)
    additional = sample_balanced(
        [r for r in records if r["title"] in title_sets["train"] and r["id"] not in seen],
        size - len(existing_train), rng,
    )
    expanded = copy.deepcopy(existing_train) + additional
    rng.shuffle(expanded)
    expanded_manifest = copy.deepcopy(manifest)
    expanded_manifest["counts"]["train"] = {
        "questions": len(expanded), "answerable": size // 2, "unanswerable": size // 2,
    }
    expanded_manifest["expanded_from"] = TRAIN_PATH
    expanded_manifest["expansion_seed"] = seed
    return expanded, expanded_manifest


def main(expand_train_size=None, output_path=None):
    with open(DEV_PATH, encoding="utf-8") as f:
        dev = json.load(f)
    dataset = load_dataset(DATASET)
    records = list(dataset["train"])
    validation_titles = set(dataset["validation"]["title"])
    if not {record["title"] for record in dev} <= validation_titles:
        raise ValueError("Existing dev set contains titles outside validation.")
    if {record["id"] for record in dev} & {record["id"] for record in records}:
        raise ValueError("Existing dev set shares question IDs with training data.")
    if expand_train_size is not None:
        path = output_path or os.path.join("data", f"train_{expand_train_size}.json")
        manifest_path = os.path.splitext(path)[0] + "_manifest.json"
        for target in (path, manifest_path):
            if os.path.exists(target):
                raise FileExistsError(f"Refusing to overwrite existing expanded data: {target}")
        frozen_paths = (TRAIN_PATH, TUNING_PATH, DEV_PATH, MANIFEST_PATH)
        frozen_hashes = {p: file_sha256(p) for p in frozen_paths}
        with open(TRAIN_PATH, encoding="utf-8") as f:
            existing_train = json.load(f)
        with open(TUNING_PATH, encoding="utf-8") as f:
            tuning = json.load(f)
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            manifest = json.load(f)
        if set(manifest["titles"]["validation"]) != validation_titles:
            raise ValueError("Existing validation title assignments differ from source dataset.")
        train, expanded_manifest = expand_training(
            records, existing_train, tuning, dev, manifest, expand_train_size
        )
        expanded_manifest["training_file"] = path
        expanded_manifest["frozen_input_sha256"] = frozen_hashes
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        for target, contents in ((path, train), (manifest_path, expanded_manifest)):
            with open(target, "x", encoding="utf-8") as f:
                json.dump(contents, f, ensure_ascii=False, indent=2)
            print(f"Wrote {target}")
        if any(file_sha256(p) != value for p, value in frozen_hashes.items()):
            raise RuntimeError("A frozen input changed while expanding the training set.")
        print(f"Expanded to {len(train)} questions ({len(train) // 2} per answerability group).")
        print("Original training, tuning, dev, and title manifest files remain unchanged.")
        return
    train, tuning, manifest = make_splits(records, validation_titles)

    os.makedirs("data", exist_ok=True)
    for path, contents in (
        (TRAIN_PATH, train),
        (TUNING_PATH, tuning),
        (MANIFEST_PATH, manifest),
    ):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(contents, f, ensure_ascii=False, indent=2)
        print(f"Wrote {path}")
    for name, counts in manifest["counts"].items():
        print(
            f"{name}: {len(manifest['titles'][name])} assigned titles, "
            f"{counts['questions']} questions "
            f"({counts['answerable']} answerable, "
            f"{counts['unanswerable']} unanswerable)"
        )
    print("Title sets are pairwise disjoint; existing dev set was not modified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expand-train-size", type=int,
                        help="Expand the existing balanced training set without resampling holdouts")
    parser.add_argument("--output-path", help="New expanded training JSON path")
    args = parser.parse_args()
    if args.output_path and args.expand_train_size is None:
        parser.error("--output-path requires --expand-train-size")
    if args.expand_train_size is not None and (args.expand_train_size <= 0 or args.expand_train_size % 2):
        parser.error("--expand-train-size must be a positive even number")
    main(args.expand_train_size, args.output_path)
