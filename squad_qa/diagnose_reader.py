"""Separate span extraction and abstention errors on tuning, without changing the model."""
import argparse
import math
import os
import time

import torch

from squad_qa import squad2_metrics as official
from squad_qa.io import load_json, save_json
from squad_qa.prepare_lora_data import TUNING_PATH, file_sha256
from squad_qa.retrieval import retrieval_contexts
from squad_qa.roberta_qa import predict, score_predictions, validate_records, weights_filename
from squad_qa.run_roberta import load_calibrated_reader

BUCKETS = (
    "correct_answered", "correct_abstained", "incorrect_answered", "incorrect_abstained",
)


def answerable_summary(rows):
    count = len(rows)
    buckets = {name: sum(row["answerable_bucket"] == name for row in rows) for name in BUCKETS}
    return {
        "questions": count,
        "before_abstention": {
            "exact": 100 * sum(row["raw_exact"] for row in rows) / count if count else None,
            "f1": 100 * sum(row["raw_f1"] for row in rows) / count if count else None,
        },
        "after_abstention": {
            "exact": 100 * sum(row["final_exact"] for row in rows) / count if count else None,
            "f1": 100 * sum(row["final_f1"] for row in rows) / count if count else None,
        },
        "buckets": buckets,
        "abstained": sum(row["abstained"] for row in rows),
        "partial_overlap_candidates": sum(0 < row["raw_f1"] < 1 for row in rows),
        "zero_overlap_candidates": sum(row["raw_f1"] == 0 for row in rows),
    }


def analyze_predictions(records, candidates, threshold, gold_retrieved, condition):
    if not math.isfinite(threshold):
        raise ValueError("Diagnostic threshold must be finite")
    if not records or len(records) != len(candidates) or len(records) != len(gold_retrieved):
        raise ValueError("Each question requires one candidate and one retrieval flag")
    before, raw_map = score_predictions(records, candidates, math.inf)
    after, final_map = score_predictions(records, candidates, threshold)
    examples = []
    for record, candidate, hit in zip(records, candidates, gold_retrieved):
        if not math.isfinite(candidate["na_score"]):
            raise ValueError(f"Non-finite no-answer score: {record['id']}")
        if not candidate["answer"] or candidate["answer"] != candidate["passage"][
            candidate["start_char"]:candidate["end_char"]
        ]:
            raise ValueError(f"Candidate does not match its source span: {record['id']}")
        qid = record["id"]
        gold = [text for text in record["answers"]["text"] if official.normalize_answer(text)] or [""]
        raw_exact = max(official.compute_exact(text, raw_map[qid]) for text in gold)
        raw_f1 = max(official.compute_f1(text, raw_map[qid]) for text in gold)
        final_exact = max(official.compute_exact(text, final_map[qid]) for text in gold)
        final_f1 = max(official.compute_f1(text, final_map[qid]) for text in gold)
        abstained = candidate["na_score"] > threshold
        answerable = bool(record["answers"]["text"])
        bucket = None
        if answerable:
            bucket = ("correct" if raw_exact else "incorrect") + (
                "_abstained" if abstained else "_answered"
            )
        examples.append({
            "id": qid, "condition": condition, "question": record["question"],
            "gold_answers": record["answers"]["text"], "answerable": answerable,
            "gold_retrieved_top3": hit, "gold_context_in_input": condition == "oracle" or hit,
            "candidate_from_gold_context": candidate["passage"] == record["context"],
            **candidate,
            "final_prediction": final_map[qid], "abstained": abstained,
            "raw_exact": raw_exact, "raw_f1": raw_f1,
            "final_exact": final_exact, "final_f1": final_f1, "answerable_bucket": bucket,
        })
    answerable_rows = [row for row in examples if row["answerable"]]
    answerable = answerable_summary(answerable_rows)
    unanswerable_rows = [row for row in examples if not row["answerable"]]
    summary = {
        "questions": len(records), "metrics_before_abstention": before,
        "metrics_after_abstention": after, "answerable": answerable,
        "answerable_by_gold_retrieved": {
            name: answerable_summary([
                row for row in answerable_rows if row["gold_retrieved_top3"] == hit
            ]) for name, hit in (("present", True), ("missing", False))
        },
        "unanswerable": {
            "questions": len(unanswerable_rows),
            "abstained": sum(row["abstained"] for row in unanswerable_rows),
            "returned_span": sum(not row["abstained"] for row in unanswerable_rows),
        },
    }
    return summary, examples


def run(args):
    started = time.perf_counter()
    output_dir = args.output_dir or os.path.join(args.adapter, "tuning_diagnostics")
    os.makedirs(output_dir, exist_ok=False)
    config_path = os.path.join(args.adapter, "qa_config.json")
    checkpoint_config = load_json(config_path)
    training_mode = checkpoint_config.get("training_mode", "lora")
    names = [weights_filename(training_mode), "qa_config.json"]
    if training_mode == "lora":
        names.append("adapter_config.json")
    paths = [TUNING_PATH] + [os.path.join(args.adapter, name) for name in names]
    hashes = {path: file_sha256(path) for path in paths}
    records = load_json(TUNING_PATH)
    validate_records(records)
    contexts, recall = retrieval_contexts(records, "tuning", args.device)
    hits = [record["context"] in passages for record, passages in zip(records, contexts)]
    model, tokenizer, config = load_calibrated_reader(args.adapter, args.device)
    model.requires_grad_(False)
    threshold = config["threshold"]
    report = {
        "adapter": args.adapter, "source": TUNING_PATH, "threshold": threshold,
        "threshold_retuned": False, "weights_updated": False,
        "training_mode": training_mode,
        "input_sha256": hashes, "retrieval": recall,
        "bucket_definition": "Exact match against any normalized reference; F1 retained separately.",
        "retrieval_caveat": (
            "Missing the gold paragraph does not prove that no retrieved passage can answer the question."
        ),
        "conditions": {},
    }
    examples = {}
    for condition, given in (
        ("retrieved", contexts), ("oracle", [[r["context"]] for r in records]),
    ):
        candidates = predict(model, tokenizer, records, given, args.batch_size)
        summary, rows = analyze_predictions(records, candidates, threshold, hits, condition)
        report["conditions"][condition] = summary
        examples[condition] = rows
        answerable = summary["answerable"]
        print(f"{condition}: answerable F1 "
              f"{answerable['before_abstention']['f1']:.2f} before -> "
              f"{answerable['after_abstention']['f1']:.2f} after abstention; "
              f"buckets={answerable['buckets']}", flush=True)
    if any(file_sha256(path) != value for path, value in hashes.items()):
        raise RuntimeError("Checkpoint or tuning data changed during diagnosis")
    report["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    save_json(os.path.join(output_dir, "summary.json"), report)
    save_json(os.path.join(output_dir, "examples.json"), examples)
    print(f"Saved fixed-threshold tuning diagnostics to {output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", help="New diagnostic directory; default is inside the adapter")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", choices=["cpu", "cuda"],
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    run(args)


if __name__ == "__main__":
    main()
