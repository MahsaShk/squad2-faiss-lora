"""Draw the fixed Gate 1 dev set: 100 answerable and 100 unanswerable questions from the validation split."""
import json
import os
import random

from datasets import load_dataset

SEED = 42
PER_GROUP = 100
OUT_PATH = os.path.join("data", "dev_200.json")


def main():
    os.makedirs("data", exist_ok=True)
    val = load_dataset("rajpurkar/squad_v2", split="validation")
    has_ans = [ex for ex in val if ex["answers"]["text"]]
    no_ans = [ex for ex in val if not ex["answers"]["text"]]
    rng = random.Random(SEED)
    sample = rng.sample(has_ans, PER_GROUP) + rng.sample(no_ans, PER_GROUP)
    records = [
        {
            "id": ex["id"],
            "title": ex["title"],
            "context": ex["context"],
            "question": ex["question"],
            "answers": {
                "text": list(ex["answers"]["text"]),
                "answer_start": list(ex["answers"]["answer_start"]),
            },
        }
        for ex in sample
    ]
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(records)} dev questions to {OUT_PATH}")


if __name__ == "__main__":
    main()
