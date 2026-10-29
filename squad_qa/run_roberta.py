"""Train, evaluate, or inspect RoBERTa QA with LoRA or a frozen-encoder span head."""
import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer

from squad_qa.io import load_json, save_json
from squad_qa.prepare_lora_data import DEV_PATH, MANIFEST_PATH, TRAIN_PATH, TUNING_PATH, file_sha256
from squad_qa.retrieval import TOP_K, retrieval_contexts
from squad_qa.mlflow_tracking import (
    add_tracking_arguments, log_epoch, log_evaluation, log_training_config,
    log_training_result, optional_run,
)
from squad_qa.roberta_qa import (
    DOC_STRIDE, MAX_ANSWER_LENGTH, MAX_LENGTH, MODEL_NAME, collate_features,
    load_reader, make_features, predict, save_reader, score_predictions, tune_threshold,
    validate_records, weights_filename,
)

SEED = 42
OUTPUT_DIR = os.path.join("outputs", "roberta")


@dataclass
class EarlyStopping:
    minimum_epochs: int
    patience: int
    min_delta: float
    best_f1: float = -math.inf
    stale_epochs: int = 0

    def update(self, epoch, f1):
        if not math.isfinite(f1):
            raise ValueError("Early stopping requires finite tuning F1")
        if f1 > self.best_f1 and f1 >= self.best_f1 + self.min_delta:
            self.best_f1 = f1
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
        return epoch >= self.minimum_epochs and self.stale_epochs >= self.patience


def check_splits(train, tuning, manifest):
    for records in (train, tuning):
        validate_records(records)
    title_sets = {name: set(manifest["titles"][name])
                  for name in ("train", "tuning", "validation")}
    if (title_sets["train"] & title_sets["tuning"]
            or title_sets["train"] & title_sets["validation"]
            or title_sets["tuning"] & title_sets["validation"]):
        raise ValueError("Split manifest has overlapping titles")
    for name, records in (("train", train), ("tuning", tuning)):
        if any(r["title"] not in title_sets[name] for r in records):
            raise ValueError(f"Question outside assigned {name} titles")
    if {r["id"] for r in train} & {r["id"] for r in tuning}:
        raise ValueError("Training and tuning question IDs overlap")


def train(args, device):
    with optional_run(args, os.path.basename(os.path.normpath(args.output_dir)), "training") as run:
        _train(args, device, run)


def _train(args, device, tracking):
    started = time.perf_counter()
    max_epochs = args.max_epochs if args.max_epochs is not None else args.epochs
    if os.path.exists(os.path.join(args.output_dir, "selection.json")) or any(
        os.path.exists(os.path.join(args.output_dir, f"epoch{epoch}"))
        for epoch in range(1, max_epochs + 1)
    ):
        raise FileExistsError("Output directory already contains training checkpoints or selection.")
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    train_records, tuning_records = load_json(args.train_path), load_json(TUNING_PATH)
    check_splits(train_records, tuning_records, load_json(MANIFEST_PATH))
    if args.limit:
        train_records = train_records[:args.limit]
    tuning_contexts, recall = retrieval_contexts(tuning_records, "tuning", device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    features = make_features(tokenizer, train_records, [[r["context"]] for r in train_records])
    positive_records = {f.record_index for f in features if f.start_position != f.cls_index}
    for index, record in enumerate(train_records):
        if record["answers"]["text"] and index not in positive_records:
            raise ValueError(f"No window contains an annotated answer: {record['id']}")
    model = load_reader(train=True, device=device, training_mode=args.training_mode)
    model.config.use_cache = False
    if args.training_mode == "lora":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.print_trainable_parameters()
    trainable = [p for p in model.parameters() if p.requires_grad]
    if args.training_mode == "head_only":
        print(f"Frozen encoder; trainable span-head parameters: "
              f"{sum(p.numel() for p in trainable):,}", flush=True)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    steps = math.ceil(math.ceil(len(features) / args.micro_batch) / args.accum) * args.epochs
    warmup = max(1, int(steps * 0.1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warmup)
    )
    amp_dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device == "cuda" and amp_dtype == torch.float16, init_scale=1024
    )
    history = {
        "base_model": MODEL_NAME, "seed": SEED, "train_questions": len(train_records),
        "train_path": args.train_path,
        "input_sha256": {p: file_sha256(p) for p in
                         (args.train_path, TUNING_PATH, MANIFEST_PATH)},
        "train_features": len(features), "training_context": "gold paragraph",
        "max_length": MAX_LENGTH, "stride": DOC_STRIDE, "max_answer_length": MAX_ANSWER_LENGTH,
        "training_mode": args.training_mode,
        "encoder_mode": "eval" if args.training_mode == "head_only" else "train",
        "lora_enabled": args.training_mode == "lora",
        "lora": ({"r": 16, "alpha": 32, "dropout": 0.05, "targets": ["query", "value"],
                  "modules_to_save": ["qa_outputs"]} if args.training_mode == "lora" else {}),
        "lr": args.lr, "micro_batch": args.micro_batch, "accum": args.accum,
        "precision": str(amp_dtype) if device == "cuda" else "torch.float32",
        "tuning_retrieval": recall, "epochs": [],
        "initial_epochs": args.epochs, "max_epochs": max_epochs,
        "early_stopping": {"patience": args.patience, "min_delta": args.min_delta,
                           "metric": "tuning overall F1", "stale_epochs": 0},
        "status": "running", "stop_reason": None,
        "trainable_parameters": sum(p.numel() for p in trainable),
    }
    if tracking is not None:
        history["mlflow_run_id"] = tracking.run_id
        log_training_config(tracking, history)
    stopper = EarlyStopping(args.epochs, args.patience, args.min_delta)
    save_json(os.path.join(args.output_dir, "selection.json"), history)
    print(f"Training mode={args.training_mode}; "
          f"{len(train_records)} questions / {len(features)} windows; "
          f"initial epochs={args.epochs}, cap={max_epochs}, patience={args.patience}, "
          f"min_delta={args.min_delta} tuning F1 points.", flush=True)
    for epoch in range(1, max_epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        if args.training_mode == "head_only":
            model.roberta.eval()
        rng.shuffle(features)
        total_loss = 0.0
        optimizer_steps = 0
        for start in range(0, len(features), args.micro_batch * args.accum):
            window = features[start:start + args.micro_batch * args.accum]
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, len(window), args.micro_batch):
                batch = window[offset:offset + args.micro_batch]
                with torch.autocast(device, dtype=amp_dtype, enabled=device == "cuda"):
                    loss = model(**collate_features(
                        batch, tokenizer.pad_token_id, device, labels=True
                    )).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
                scaler.scale(loss * len(batch) / len(window)).backward()
                total_loss += loss.item() * len(batch)
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if not torch.isfinite(grad_norm):
                if not scaler.is_enabled():
                    raise FloatingPointError(f"Non-finite gradients at epoch {epoch}")
                print(f"Warning: AMP skipped an overflowing update at epoch {epoch}; "
                      "reducing the loss scale.", flush=True)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                scheduler.step()
                optimizer_steps += 1
                if optimizer_steps % args.log_every == 0:
                    processed = start + len(window)
                    print(f"epoch {epoch}/{max_epochs}: windows {processed}/{len(features)}, "
                          f"updates {optimizer_steps}, loss {total_loss / processed:.4f}, "
                          f"elapsed {time.perf_counter() - epoch_started:.0f}s", flush=True)
        if not optimizer_steps:
            raise FloatingPointError(f"No successful optimizer updates at epoch {epoch}")
        path = os.path.join(args.output_dir, f"epoch{epoch}")
        save_reader(model, path, args.training_mode)
        tokenizer.save_pretrained(path)
        rows = predict(model, tokenizer, tuning_records, tuning_contexts, args.batch_size)
        threshold, metrics = tune_threshold(tuning_records, rows)
        calibration = {
            "base_model": MODEL_NAME, "threshold": threshold, "tuned_on": TUNING_PATH,
            "context": "retrieved top-3", "max_length": MAX_LENGTH, "stride": DOC_STRIDE,
            "max_answer_length": MAX_ANSWER_LENGTH, "top_k": TOP_K, "metrics": metrics,
            "training_mode": args.training_mode,
        }
        save_json(os.path.join(path, "qa_config.json"), calibration)
        entry = {"epoch": epoch, "adapter": path, "loss": total_loss / len(features),
                 "optimizer_steps": optimizer_steps,
                 "threshold": threshold, "tuning": metrics,
                 "elapsed_seconds": round(time.perf_counter() - epoch_started, 2)}
        history["epochs"].append(entry)
        best = max(history["epochs"], key=lambda e: e["tuning"]["all"]["f1"])
        history["selected_adapter"] = best["adapter"]
        history["selected_epoch"] = best["epoch"]
        stop = stopper.update(epoch, metrics["all"]["f1"])
        history["early_stopping"]["stale_epochs"] = stopper.stale_epochs
        history["completed_epochs"] = epoch
        history["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        if epoch == max_epochs or stop:
            history["status"] = "completed"
            history["stop_reason"] = "max_epochs" if epoch == max_epochs else "early_stopping"
        save_json(os.path.join(args.output_dir, "selection.json"), history)
        if tracking is not None:
            log_epoch(tracking, entry)
        print(json.dumps(entry, indent=2), flush=True)
        print(f"epoch {epoch}: tuning overall F1={metrics['all']['f1']:.2f}, "
              f"answerable F1={metrics['hasans']['f1']:.2f}, "
              f"unanswerable accuracy={metrics['noans']['exact']:.2f}%; "
              f"stale epochs={stopper.stale_epochs}/{args.patience}", flush=True)
        if stop:
            print(f"Early stopping after {epoch} epochs.", flush=True)
            break
    print(f"Training finished: {history['stop_reason']}; "
          f"{history['completed_epochs']} epochs.", flush=True)
    print(f"Selected adapter: {history['selected_adapter']}", flush=True)
    if tracking is not None:
        log_training_result(
            tracking, history, os.path.join(args.output_dir, "selection.json"),
            log_adapter=args.mlflow_log_adapter,
        )


def load_calibrated_reader(adapter, device):
    config = load_json(os.path.join(adapter, "qa_config.json"))
    expected = {"base_model": MODEL_NAME, "max_length": MAX_LENGTH, "stride": DOC_STRIDE,
                "max_answer_length": MAX_ANSWER_LENGTH, "top_k": TOP_K,
                "context": "retrieved top-3"}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Adapter calibration does not match reader/retrieval settings")
    if not math.isfinite(config["threshold"]):
        raise ValueError("Adapter threshold must be finite")
    tokenizer = AutoTokenizer.from_pretrained(adapter, use_fast=True)
    training_mode = config.get("training_mode", "lora")
    weights_filename(training_mode)
    return load_reader(
        adapter=adapter, device=device, training_mode=training_mode
    ), tokenizer, config


def evaluate(args, device):
    with optional_run(args, "roberta-evaluation", "evaluation") as run:
        _evaluate(args, device, run)


def _evaluate(args, device, tracking):
    records = load_json(DEV_PATH)
    validate_records(records)
    contexts, recall = retrieval_contexts(records, "dev", device)
    model, tokenizer, config = load_calibrated_reader(args.adapter, device)
    results = {"adapter": args.adapter, "config": config, "retrieval": recall}
    predictions, details = {}, {}
    for name, given in (
        ("retrieved", contexts), ("oracle", [[r["context"]] for r in records]),
    ):
        rows = predict(model, tokenizer, records, given, args.batch_size)
        results[name], predictions[name] = score_predictions(records, rows, config["threshold"])
        details[name] = [{"id": record["id"], **row} for record, row in zip(records, rows)]
    save_json(os.path.join(args.adapter, "dev_results.json"), results)
    save_json(os.path.join(args.adapter, "dev_predictions.json"), predictions)
    save_json(os.path.join(args.adapter, "dev_candidates.json"), details)
    if tracking is not None:
        log_evaluation(tracking, results, os.path.join(args.adapter, "dev_results.json"))
    print(json.dumps(results, indent=2))


def ask(args, device):
    if args.question:
        record = {"id": "custom", "question": args.question, "answers": {
            "text": [], "answer_start": [],
        }}
    else:
        records = load_json(DEV_PATH)
        if args.id:
            matches = [r for r in records if r["id"] == args.id]
            if not matches:
                raise ValueError(f"Unknown dev question ID: {args.id}")
            record = matches[0]
        else:
            if not 0 <= args.index < len(records):
                raise ValueError("Dev question index is out of range")
            record = records[args.index]
    contexts, _ = retrieval_contexts([record], "dev", device)
    model, tokenizer, config = load_calibrated_reader(args.adapter, device)
    row = predict(model, tokenizer, [record], contexts, args.batch_size)[0]
    prediction = "" if row["na_score"] > config["threshold"] else row["answer"]
    print(f"Question: {record['question']}")
    for rank, passage in enumerate(contexts[0], 1):
        print(f"\n[{rank}] {passage}")
    print(f"\nBest span: {row['answer']!r} ({row['start_char']}:{row['end_char']})")
    print(f"No-answer margin: {row['na_score']:.4f}; threshold: {config['threshold']:.4f}")
    print(f"Prediction: {prediction or '(abstain)'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"],
                        default="cuda" if torch.cuda.is_available() else "cpu")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--training-mode", choices=["lora", "head_only"], default="lora",
                          help="Use LoRA, or freeze RoBERTa and train only the span head")
    training.add_argument("--epochs", type=int, default=3,
                          help="Initial epoch budget; also the minimum before early stopping")
    training.add_argument("--max-epochs", type=int,
                          help="Optional extension cap; default is --epochs (fixed-length run)")
    training.add_argument("--patience", type=int, default=3)
    training.add_argument("--min-delta", type=float, default=0.1,
                          help="Minimum tuning F1 improvement in percentage points")
    training.add_argument("--train-path", default=TRAIN_PATH)
    training.add_argument("--log-every", type=int, default=50,
                          help="Log training progress every N successful optimizer updates")
    training.add_argument("--micro-batch", type=int, default=2)
    training.add_argument("--accum", type=int, default=8)
    training.add_argument("--lr", type=float, default=2e-4)
    training.add_argument("--limit", type=int, help="Limit training questions for a smoke run")
    training.add_argument("--output-dir", default=OUTPUT_DIR)
    evaluation = commands.add_parser("evaluate")
    add_tracking_arguments(training)
    training.add_argument("--mlflow-log-adapter", action="store_true",
                          help="Copy only the selected checkpoint/tokenizer to local MLflow artifacts")
    add_tracking_arguments(evaluation, resume=True)
    single = commands.add_parser("ask")
    for command in (training, evaluation, single):
        command.add_argument("--batch-size", type=int, default=8, help="Reader inference batch size")
    for command in (evaluation, single):
        command.add_argument("--adapter", required=True, help="Epoch directory with qa_config.json")
    questions = single.add_mutually_exclusive_group()
    questions.add_argument("--question")
    questions.add_argument("--id")
    questions.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.command == "train":
        if args.epochs <= 0 or args.micro_batch <= 0 or args.accum <= 0:
            parser.error("--epochs, --micro-batch, and --accum must be positive")
        if not math.isfinite(args.lr) or args.lr <= 0 or (args.limit is not None and args.limit <= 0):
            parser.error("--lr and --limit must be positive (and learning rate finite)")
        if args.max_epochs is not None and args.max_epochs < args.epochs:
            parser.error("--max-epochs must be at least --epochs")
        if args.patience <= 0 or args.log_every <= 0:
            parser.error("--patience and --log-every must be positive")
        if not math.isfinite(args.min_delta) or args.min_delta < 0:
            parser.error("--min-delta must be finite and non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    {"train": train, "evaluate": evaluate, "ask": ask}[args.command](args, args.device)


if __name__ == "__main__":
    main()
