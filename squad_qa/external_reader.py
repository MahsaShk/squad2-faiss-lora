"""Evaluate an off-the-shelf SQuAD2 reader with existing FAISS retrieval; no training."""
import argparse
import os
import time

import torch
from huggingface_hub import model_info
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from squad_qa.io import load_json, save_json
from squad_qa.prepare_lora_data import DEV_PATH, TUNING_PATH
from squad_qa.roberta_qa import (
    DOC_STRIDE, MAX_ANSWER_LENGTH, MAX_LENGTH, predict, score_predictions,
    tune_threshold, validate_records,
)
from squad_qa.retrieval import TOP_K, retrieval_contexts
from squad_qa.mlflow_tracking import add_tracking_arguments, log_external_result, optional_run

MODEL_NAME = "deepset/roberta-base-squad2"
OUTPUT_DIR = os.path.join("outputs", "external_reader")
EXTERNAL_DATA_CAVEAT = (
    "This checkpoint was already fine-tuned on SQuAD2 outside this project. "
    "Its training may include our custom tuning questions from SQuAD's training split; "
    "it does not follow our title-disjoint model-training protocol. "
    "SQuAD validation has also been used for published model evaluation."
)


def run(args):
    with optional_run(args, "external-reader", "external evaluation") as tracking:
        _run(args, tracking)


def _run(args, tracking):
    started = time.perf_counter()
    os.makedirs(args.output_dir, exist_ok=False)
    print(f"External reader: {MODEL_NAME}; no training or LoRA.", flush=True)
    print(f"Evaluation caveat: {EXTERNAL_DATA_CAVEAT}", flush=True)
    tuning, dev = load_json(TUNING_PATH), load_json(DEV_PATH)
    validate_records(tuning)
    validate_records(dev)
    if {r["id"] for r in tuning} & {r["id"] for r in dev}:
        raise ValueError("Tuning and dev question IDs overlap")
    if {r["title"] for r in tuning} & {r["title"] for r in dev}:
        raise ValueError("Tuning and dev titles overlap")
    tuning_contexts, tuning_recall = retrieval_contexts(tuning, "tuning", args.device)
    dev_contexts, dev_recall = retrieval_contexts(dev, "dev", args.device)

    revision = model_info(MODEL_NAME).sha
    if not revision:
        raise ValueError("External checkpoint metadata did not provide a revision")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=revision, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(
        MODEL_NAME, revision=revision
    ).to(args.device)
    model.requires_grad_(False)
    model.eval()
    tuning_rows = predict(model, tokenizer, tuning, tuning_contexts, args.batch_size)
    threshold, tuning_metrics = tune_threshold(tuning, tuning_rows)
    zero_tuning_metrics, _ = score_predictions(tuning, tuning_rows, 0.0)
    config = {
        "reader_model": MODEL_NAME,
        "model_revision": revision,
        "trained_in_this_run": False, "lora": False,
        "device": args.device, "batch_size": args.batch_size,
        "max_length": MAX_LENGTH, "stride": DOC_STRIDE,
        "max_answer_length": MAX_ANSWER_LENGTH, "top_k": TOP_K,
        "aggregation": "smallest window-local CLS-minus-best-span margin",
        "external_data_caveat": EXTERNAL_DATA_CAVEAT,
    }
    calibration = {
        "source": TUNING_PATH, "context": "retrieved top-3",
        "threshold": threshold, "selection": "tuning overall F1",
        "zero_margin_threshold": 0.0, "zero_margin_metrics": zero_tuning_metrics,
        "calibrated_metrics": tuning_metrics,
    }
    save_json(os.path.join(args.output_dir, "calibration.json"), {
        "config": config, **calibration,
    })
    print(f"Tuning complete: threshold={threshold:.6f}; "
          f"F1={tuning_metrics['all']['f1']:.2f}", flush=True)

    results = {
        "config": config, "calibration": calibration,
        "retrieval": {"tuning": tuning_recall, "dev": dev_recall},
        "dev": {
            "zero_margin": {"threshold": 0.0},
            "calibrated": {"threshold": threshold},
        },
    }
    predictions = {"zero_margin": {}, "calibrated": {}}
    candidates = {
        "tuning": [{"id": r["id"], **row} for r, row in zip(tuning, tuning_rows)],
    }
    for condition, contexts in (
        ("retrieved", dev_contexts), ("oracle", [[r["context"]] for r in dev]),
    ):
        rows = predict(model, tokenizer, dev, contexts, args.batch_size)
        candidates[condition] = [{"id": r["id"], **row} for r, row in zip(dev, rows)]
        for mode, value in (("zero_margin", 0.0), ("calibrated", threshold)):
            metrics, answer_map = score_predictions(dev, rows, value)
            results["dev"][mode][condition] = metrics
            predictions[mode][condition] = answer_map
        print(f"Dev {condition}: zero-margin F1="
              f"{results['dev']['zero_margin'][condition]['all']['f1']:.2f}; calibrated F1="
              f"{results['dev']['calibrated'][condition]['all']['f1']:.2f}", flush=True)

    results["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    save_json(os.path.join(args.output_dir, "results.json"), results)
    save_json(os.path.join(args.output_dir, "predictions.json"), predictions)
    save_json(os.path.join(args.output_dir, "candidates.json"), candidates)
    if tracking is not None:
        log_external_result(tracking, results, os.path.join(args.output_dir, "results.json"))
    print(f"Saved external-reader results to {args.output_dir} "
          f"({results['elapsed_seconds']:.1f} seconds)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_tracking_arguments(parser)
    parser.add_argument("--device", choices=["cpu", "cuda"],
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help="New output directory; existing directories are not overwritten")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    run(args)


if __name__ == "__main__":
    main()
