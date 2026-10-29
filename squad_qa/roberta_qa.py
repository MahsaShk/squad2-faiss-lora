"""Extractive RoBERTa QA: sliding windows, span supervision, and abstention."""
import math
import os
from dataclasses import dataclass

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from safetensors.torch import load_file, save_file
from transformers import AutoModelForQuestionAnswering

from squad_qa import squad2_metrics as official

MODEL_NAME = "FacebookAI/roberta-base"
MAX_LENGTH = 384
DOC_STRIDE = 128
MAX_ANSWER_LENGTH = 30


@dataclass
class Feature:
    record_index: int
    passage: str
    input_ids: list[int]
    attention_mask: list[int]
    offsets: list[tuple[int, int] | None]
    cls_index: int
    start_position: int
    end_position: int


def validate_records(records):
    if not records:
        raise ValueError("Question records must not be empty")
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate question IDs")
    for r in records:
        texts, starts = r["answers"]["text"], r["answers"]["answer_start"]
        if len(texts) != len(starts):
            raise ValueError(f"Mismatched answer texts and offsets: {r['id']}")
        for text, start in zip(texts, starts):
            if not text or start < 0 or r["context"][start:start + len(text)] != text:
                raise ValueError(f"Answer does not match context offset: {r['id']}")


def span_positions(offsets, answers, cls_index):
    """Use CLS when no annotated answer is fully contained in this window."""
    context_tokens = [
        i for i, offset in enumerate(offsets)
        if offset is not None and offset[1] > offset[0]
    ]
    if not context_tokens:
        raise ValueError("Tokenized window contains no context")
    first, last = context_tokens[0], context_tokens[-1]
    for text, start in zip(answers["text"], answers["answer_start"]):
        end = start + len(text)
        if offsets[first][0] <= start and offsets[last][1] >= end:
            left = next(i for i in context_tokens if offsets[i][1] > start)
            right = next(i for i in reversed(context_tokens) if offsets[i][0] < end)
            if left <= right:
                return left, right
    return cls_index, cls_index


def make_features(tokenizer, records, contexts, max_length=MAX_LENGTH, stride=DOC_STRIDE):
    if not tokenizer.is_fast or tokenizer.padding_side != "right":
        raise ValueError("QA requires a fast tokenizer with right padding")
    if len(records) != len(contexts) or not records:
        raise ValueError("Each question must have a matching context list")
    if not 0 <= stride < max_length <= 512:
        raise ValueError("Require 0 <= stride < max_length <= 512")
    features = []
    for record_index, (record, passages) in enumerate(zip(records, contexts)):
        if not passages:
            raise ValueError(f"No passages for question {record['id']}")
        question = record["question"].strip()
        question_length = len(tokenizer(question, add_special_tokens=False)["input_ids"])
        capacity = max_length - question_length - tokenizer.num_special_tokens_to_add(pair=True)
        if capacity <= stride:
            raise ValueError(f"Question leaves too little room for context: {record['id']}")
        for passage in passages:
            if not passage.strip():
                raise ValueError(f"Empty passage for question {record['id']}")
            encoded = tokenizer(
                question, passage, truncation="only_second", max_length=max_length,
                stride=stride, return_overflowing_tokens=True, return_offsets_mapping=True,
            )
            answers = record["answers"] if passage == record.get("context") else {
                "text": [], "answer_start": [],
            }
            for window, ids in enumerate(encoded["input_ids"]):
                offsets = [
                    tuple(offset) if sequence == 1 else None
                    for sequence, offset in zip(
                        encoded.sequence_ids(window), encoded["offset_mapping"][window]
                    )
                ]
                cls_index = ids.index(tokenizer.cls_token_id)
                start, end = span_positions(offsets, answers, cls_index)
                features.append(Feature(
                    record_index, passage, ids, encoded["attention_mask"][window],
                    offsets, cls_index, start, end,
                ))
    return features


def collate_features(features, pad_id, device, labels=False):
    width = max(len(f.input_ids) for f in features)
    ids = torch.full((len(features), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for row, feature in enumerate(features):
        length = len(feature.input_ids)
        ids[row, :length] = torch.tensor(feature.input_ids, device=device)
        mask[row, :length] = torch.tensor(feature.attention_mask, device=device)
    batch = {"input_ids": ids, "attention_mask": mask}
    if labels:
        batch["start_positions"] = torch.tensor(
            [f.start_position for f in features], device=device
        )
        batch["end_positions"] = torch.tensor(
            [f.end_position for f in features], device=device
        )
    return batch


def add_lora(model):
    return get_peft_model(model, LoraConfig(
        task_type=TaskType.QUESTION_ANS, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["query", "value"], modules_to_save=["qa_outputs"],
    ))


def weights_filename(training_mode):
    if training_mode == "lora":
        return "adapter_model.safetensors"
    if training_mode == "head_only":
        return "qa_head.safetensors"
    raise ValueError(f"Unknown reader training mode: {training_mode}")


def save_reader(model, path, training_mode="lora"):
    filename = weights_filename(training_mode)
    if training_mode == "lora":
        model.save_pretrained(path)
    else:
        os.makedirs(path, exist_ok=True)
        save_file({
            name: value.detach().cpu().contiguous()
            for name, value in model.qa_outputs.state_dict().items()
        }, os.path.join(path, filename), metadata={"format": "pt"})


def load_reader(adapter=None, train=False, device="cpu", training_mode="lora"):
    filename = weights_filename(training_mode)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    if training_mode == "head_only":
        model.requires_grad_(False)
        if adapter:
            model.qa_outputs.load_state_dict(load_file(os.path.join(adapter, filename)))
        model.qa_outputs.requires_grad_(train)
    elif adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=train)
    elif train:
        model = add_lora(model)
    return model.to(device)


def window_candidate(feature, start_logits, end_logits, max_answer_length):
    """Compare span and CLS within one window, avoiding cross-window logit shifts."""
    null_score = start_logits[feature.cls_index] + end_logits[feature.cls_index]
    best = None
    for start, offset in enumerate(feature.offsets):
        if offset is None or offset[1] <= offset[0]:
            continue
        for end in range(start, min(len(feature.offsets), start + max_answer_length)):
            end_offset = feature.offsets[end]
            if end_offset is None:
                break
            if end_offset[1] <= end_offset[0]:
                continue
            if end_offset[1] <= offset[0]:
                continue
            score = start_logits[start] + end_logits[end]
            margin = float(null_score - score)
            if best is None or margin < best["na_score"]:
                best = {
                    "answer": feature.passage[offset[0]:end_offset[1]],
                    "start_char": offset[0], "end_char": end_offset[1],
                    "na_score": margin, "passage": feature.passage,
                }
    if best is None or not math.isfinite(best["na_score"]):
        raise ValueError("Window has no valid finite span candidate")
    return best


@torch.no_grad()
def predict(model, tokenizer, records, contexts, batch_size=8,
            max_length=MAX_LENGTH, stride=DOC_STRIDE, max_answer_length=MAX_ANSWER_LENGTH):
    if batch_size <= 0 or max_answer_length <= 0:
        raise ValueError("Batch size and maximum answer length must be positive")
    features = make_features(tokenizer, records, contexts, max_length, stride)
    best = [None] * len(records)
    model.eval()
    device = next(model.parameters()).device
    for start in range(0, len(features), batch_size):
        batch = features[start:start + batch_size]
        outputs = model(**collate_features(batch, tokenizer.pad_token_id, device))
        starts = outputs.start_logits.float().cpu().tolist()
        ends = outputs.end_logits.float().cpu().tolist()
        for feature, start_logits, end_logits in zip(batch, starts, ends):
            candidate = window_candidate(feature, start_logits, end_logits, max_answer_length)
            previous = best[feature.record_index]
            if previous is None or candidate["na_score"] < previous["na_score"]:
                best[feature.record_index] = candidate
    return best


def score_predictions(records, rows, threshold):
    if len(records) != len(rows) or not records:
        raise ValueError("Each question must have one prediction row")
    predictions = {
        r["id"]: "" if row["na_score"] > threshold else row["answer"]
        for r, row in zip(records, rows)
    }
    dataset = [{"paragraphs": [{"qas": [
        {"id": r["id"], "answers": [{"text": t} for t in r["answers"]["text"]]}
        for r in records
    ]}]}]
    exact, f1 = official.get_raw_scores(dataset, predictions)
    metrics = {}
    for name, ids in (
        ("all", [r["id"] for r in records]),
        ("hasans", [r["id"] for r in records if r["answers"]["text"]]),
        ("noans", [r["id"] for r in records if not r["answers"]["text"]]),
    ):
        metrics[name] = official.make_eval_dict(exact, f1, ids) if ids else {
            "exact": None, "f1": None, "total": 0,
        }
    return metrics, predictions


def tune_threshold(records, rows):
    """Sweep tied margins as groups so the selected score is actually achievable."""
    score_predictions(records, rows, 0.0)
    if any(not math.isfinite(row["na_score"]) for row in rows):
        raise ValueError("Threshold tuning requires finite no-answer scores")
    ranked = sorted(zip(records, rows), key=lambda pair: pair[1]["na_score"])
    threshold = math.nextafter(ranked[0][1]["na_score"], -math.inf)
    baseline, _ = score_predictions(records, rows, threshold)
    current = baseline["all"]["f1"] * len(records) / 100.0
    best = current
    index = 0
    while index < len(ranked):
        margin = ranked[index][1]["na_score"]
        while index < len(ranked) and ranked[index][1]["na_score"] == margin:
            record, row = ranked[index]
            gold = [t for t in record["answers"]["text"] if official.normalize_answer(t)] or [""]
            before = max(official.compute_f1(text, "") for text in gold)
            after = max(official.compute_f1(text, row["answer"]) for text in gold)
            current += after - before
            index += 1
        if current > best:
            best, threshold = current, margin
    metrics, _ = score_predictions(records, rows, threshold)
    return threshold, metrics
