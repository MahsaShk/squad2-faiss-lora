"""Offline tests using a local byte-BPE tokenizer and a tiny random RoBERTa."""
import argparse
import contextlib
import copy
import io
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
from peft import PeftModel
from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizerFast

from squad_qa.roberta_qa import (
    Feature, add_lora, collate_features, make_features, predict, score_predictions,
    load_reader, save_reader, span_positions, tune_threshold, validate_records, window_candidate,
)
from squad_qa import run_roberta


def record(qid="a", context="Paris is in France.", answer="France"):
    return {
        "id": qid, "title": "Train", "context": context, "question": "Where is Paris?",
        "answers": {"text": [answer] if answer else [],
                    "answer_start": [context.index(answer)] if answer else []},
    }


def tiny_model(vocab_size):
    return RobertaForQuestionAnswering(RobertaConfig(
        vocab_size=vocab_size, hidden_size=24, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=32, max_position_embeddings=514,
        pad_token_id=1, bos_token_id=0, eos_token_id=2,
    ))


class ReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        bpe = ByteLevelBPETokenizer()
        bpe.train_from_iterator(
            ["Where is Paris? Paris is in France. " * 20,
             "start middle end target " * 20],
            vocab_size=300, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
        )
        bpe.save_model(cls.temp.name)
        cls.tokenizer = RobertaTokenizerFast(
            vocab=os.path.join(cls.temp.name, "vocab.json"),
            merges=os.path.join(cls.temp.name, "merges.txt"),
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_exact_offsets_and_cls_supervision(self):
        records = [record(), record("u", answer="")]
        validate_records(records)
        features = make_features(self.tokenizer, records, [[r["context"]] for r in records])
        first, second = features
        start = first.offsets[first.start_position][0]
        end = first.offsets[first.end_position][1]
        self.assertEqual(first.passage[start:end], "France")
        self.assertEqual((second.start_position, second.end_position),
                         (second.cls_index, second.cls_index))
        question_token = first.input_ids.index(self.tokenizer.cls_token_id)
        self.assertIsNone(first.offsets[question_token])

    def test_overflow_preserves_late_answer_and_labels_other_windows_null(self):
        r = record(context="middle " * 100 + "target", answer="target")
        features = make_features(self.tokenizer, [r], [[r["context"]]], max_length=48, stride=16)
        positives = [f for f in features if f.start_position != f.cls_index]
        self.assertGreater(len(features), 2)
        self.assertTrue(positives)
        self.assertEqual(features[0].start_position, features[0].cls_index)
        self.assertEqual(positives[-1].passage[
            positives[-1].offsets[positives[-1].start_position][0]:
            positives[-1].offsets[positives[-1].end_position][1]
        ], "target")

    def test_other_passages_are_not_assigned_gold_offsets(self):
        r = record()
        features = make_features(self.tokenizer, [r], [["Unrelated France text."]])
        self.assertTrue(all(f.start_position == f.cls_index for f in features))

    def test_multiple_answers_and_partial_window(self):
        offsets = [None, (10, 15), (16, 20)]
        answers = {"text": ["missing", "word"], "answer_start": [0, 16]}
        self.assertEqual(span_positions(offsets, answers, 0), (2, 2))
        self.assertEqual(span_positions(offsets, {"text": ["long answer"], "answer_start": [8]}, 0),
                         (0, 0))

    def test_zero_width_spaces_and_duplicate_unicode_offsets(self):
        offsets = [None, (0, 1), (1, 1), (2, 3), (2, 3), None]
        self.assertEqual(span_positions(
            offsets, {"text": ["a b"], "answer_start": [0]}, 0
        ), (1, 4))
        self.assertEqual(span_positions(
            offsets, {"text": ["b"], "answer_start": [2]}, 0
        ), (3, 4))
        feature = Feature(0, "a b", [0] * 6, [1] * 6, offsets, 0, 1, 4)
        row = window_candidate(feature, [0, 8, 100, 1, 1, 0], [0, 1, 100, 1, 9, 0], 4)
        self.assertEqual(row["answer"], "a b")

    def test_invalid_input_is_explicit(self):
        r = record()
        broken = copy.deepcopy(r)
        broken["answers"]["answer_start"] = [0]
        with self.assertRaisesRegex(ValueError, "offset"):
            validate_records([broken])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_records([r, r])
        with self.assertRaisesRegex(ValueError, "No passages"):
            make_features(self.tokenizer, [r], [[]])
        r["question"] = "middle " * 100
        with self.assertRaisesRegex(ValueError, "too little room"):
            make_features(self.tokenizer, [r], [[r["context"]]], max_length=48, stride=16)

    def test_decoder_excludes_question_special_and_padding_and_limits_span(self):
        f = Feature(0, "Paris France", [0] * 5, [1] * 5,
                    [None, None, (0, 5), (6, 12), None], 0, 0, 0)
        starts = [0, 1000, 8, 1, 2000]
        ends = [0, 1000, 1, 9, 2000]
        full = window_candidate(f, starts, ends, 2)
        self.assertEqual(full["answer"], "Paris France")
        limited = window_candidate(f, starts, ends, 1)
        self.assertEqual(limited["answer"], "France")
        self.assertEqual(limited["na_score"], -10)

    def test_threshold_sweep_respects_ties_and_strict_comparison(self):
        records = [record(), record("u", answer="")]
        rows = [{"answer": "France", "na_score": 2.0},
                {"answer": "France", "na_score": 2.0}]
        threshold, metrics = tune_threshold(records, rows)
        applied, predictions = score_predictions(records, rows, threshold)
        self.assertEqual(metrics["all"]["f1"], 50.0)
        self.assertEqual(applied, metrics)
        self.assertEqual(predictions, {"a": "", "u": ""})
        rows[0]["na_score"] = -2.0
        threshold, metrics = tune_threshold(records, rows)
        self.assertEqual(threshold, -2.0)
        self.assertEqual(metrics["all"]["f1"], 100.0)

    def test_literal_unanswerable_is_not_rewritten_as_abstention(self):
        r = record(context="unanswerable", answer="unanswerable")
        metrics, predictions = score_predictions(
            [r], [{"answer": "unanswerable", "na_score": 0.0}], 0.0
        )
        self.assertEqual(predictions["a"], "unanswerable")
        self.assertEqual(metrics["all"]["f1"], 100)
        self.assertEqual(metrics["noans"]["total"], 0)

    def test_tuning_matches_official_normalization_and_handles_negative_scores(self):
        records = [record(context="the", answer="the"), record("u", answer="")]
        rows = [{"answer": "wrong", "na_score": -5.0},
                {"answer": "wrong", "na_score": -4.0}]
        threshold, metrics = tune_threshold(records, rows)
        self.assertLess(threshold, -5)
        self.assertEqual(metrics["all"]["f1"], 100)

    def test_lora_backward_save_reload_and_multicontext_prediction(self):
        torch.manual_seed(42)
        base = tiny_model(len(self.tokenizer))
        base_state = copy.deepcopy(base.state_dict())
        model = add_lora(base)
        r = record()
        features = make_features(self.tokenizer, [r], [[r["context"]]])
        batch = collate_features(features, self.tokenizer.pad_token_id, "cpu", labels=True)
        loss = model(**batch).loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        named = dict(model.named_parameters())
        self.assertTrue(any("lora_" in n and p.grad is not None for n, p in named.items()))
        self.assertTrue(any("qa_outputs.modules_to_save" in n and p.grad is not None
                            for n, p in named.items()))
        self.assertFalse(any(p.requires_grad for n, p in named.items()
                             if "lora_" not in n and "modules_to_save" not in n))
        model.eval()
        expected = model(**batch).start_logits.detach()
        with tempfile.TemporaryDirectory() as path:
            model.save_pretrained(path)
            restored_base = tiny_model(len(self.tokenizer))
            restored_base.load_state_dict(base_state)
            restored = PeftModel.from_pretrained(restored_base, path).eval()
            torch.testing.assert_close(restored(**batch).start_logits, expected)
            contexts = [[r["context"], "Paris is elsewhere."]]
            row = predict(restored, self.tokenizer, [r], contexts, batch_size=1)[0]
            self.assertIn(row["passage"], contexts[0])
            self.assertEqual(row["answer"], row["passage"][row["start_char"]:row["end_char"]])

    def test_train_writes_selected_adapter_tokenizer_and_calibration(self):
        training = [record(), record("u", answer="")]
        tuning = [dict(record("t"), title="Tune"), dict(record("tu", answer=""), title="Tune")]
        manifest = {"titles": {"train": ["Train"], "tuning": ["Tune"], "validation": ["Dev"]}}
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                limit=None, lr=2e-4, micro_batch=2, accum=2, epochs=1,
                batch_size=2, output_dir=directory, train_path=run_roberta.TRAIN_PATH,
                max_epochs=None, patience=3, min_delta=0.1, log_every=50,
                mlflow=False,
                training_mode="lora",
            )
            fixtures = {run_roberta.TRAIN_PATH: training, run_roberta.TUNING_PATH: tuning,
                        run_roberta.MANIFEST_PATH: manifest}
            with patch.object(run_roberta, "load_json", side_effect=fixtures.__getitem__), \
                    patch.object(run_roberta, "file_sha256", return_value="fixture-hash"), \
                    patch.object(run_roberta, "retrieval_contexts",
                                 return_value=([[r["context"]] for r in tuning], {})), \
                    patch.object(run_roberta.AutoTokenizer, "from_pretrained",
                                 return_value=self.tokenizer), \
                    patch.object(run_roberta, "load_reader",
                                 return_value=add_lora(tiny_model(len(self.tokenizer)))):
                run_roberta.train(args, "cpu")
            with open(os.path.join(directory, "selection.json"), encoding="utf-8") as f:
                selection = json.load(f)
            self.assertEqual(selection["selected_epoch"], 1)
            self.assertEqual(selection["completed_epochs"], 1)
            self.assertEqual(selection["status"], "completed")
            self.assertEqual(selection["stop_reason"], "max_epochs")
            adapter = selection["selected_adapter"]
            for filename in ("adapter_model.safetensors", "tokenizer.json", "qa_config.json"):
                self.assertTrue(os.path.isfile(os.path.join(adapter, filename)))
            with open(os.path.join(adapter, "qa_config.json"), encoding="utf-8") as f:
                config = json.load(f)
            self.assertEqual(config["tuned_on"], run_roberta.TUNING_PATH)
            self.assertEqual(config["metrics"]["all"]["total"], 2)

    def test_adaptive_training_extends_then_stops_and_selects_tuning_best(self):
        training = [record(), record("u", answer="")]
        tuning = [dict(record("t"), title="Tune"), dict(record("tu", answer=""), title="Tune")]
        manifest = {"titles": {"train": ["Train"], "tuning": ["Tune"], "validation": ["Dev"]}}
        scores = [50.0, 51.0, 52.0, 53.0, 53.0, 53.0, 53.0]
        calibrated = [
            (-float(epoch), {
                "all": {"exact": score, "f1": score, "total": 2},
                "hasans": {"exact": 20.0, "f1": 25.0, "total": 1},
                "noans": {"exact": 80.0, "f1": 80.0, "total": 1},
            })
            for epoch, score in enumerate(scores, 1)
        ]
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                limit=None, lr=2e-4, micro_batch=2, accum=2, epochs=5,
                max_epochs=15, patience=3, min_delta=0.1, log_every=1,
                batch_size=2, output_dir=directory, train_path="expanded-fixture.json",
                mlflow=False,
                training_mode="lora",
            )
            fixtures = {args.train_path: training, run_roberta.TUNING_PATH: tuning,
                        run_roberta.MANIFEST_PATH: manifest}
            with patch.object(run_roberta, "load_json",
                              side_effect=fixtures.__getitem__) as load, \
                    patch.object(run_roberta, "file_sha256", return_value="fixture-hash"), \
                    patch.object(run_roberta, "retrieval_contexts",
                                 return_value=([[r["context"]] for r in tuning], {})), \
                    patch.object(run_roberta.AutoTokenizer, "from_pretrained",
                                 return_value=self.tokenizer), \
                    patch.object(run_roberta, "load_reader",
                                 return_value=add_lora(tiny_model(len(self.tokenizer)))), \
                    patch.object(run_roberta, "tune_threshold", side_effect=calibrated) as tune, \
                    contextlib.redirect_stdout(io.StringIO()):
                run_roberta.train(args, "cpu")
            self.assertEqual(tune.call_count, 7)
            self.assertNotIn(run_roberta.DEV_PATH, [call.args[0] for call in load.call_args_list])
            with open(os.path.join(directory, "selection.json"), encoding="utf-8") as f:
                selection = json.load(f)
            self.assertEqual(selection["completed_epochs"], 7)
            self.assertEqual(selection["selected_epoch"], 4)
            self.assertEqual(selection["stop_reason"], "early_stopping")
            self.assertEqual(selection["early_stopping"]["stale_epochs"], 3)
            self.assertEqual(selection["epochs"][-1]["tuning"]["hasans"]["f1"], 25)
            self.assertEqual(selection["epochs"][-1]["tuning"]["noans"]["exact"], 80)
            with open(os.path.join(selection["selected_adapter"], "qa_config.json"),
                      encoding="utf-8") as f:
                self.assertEqual(json.load(f)["threshold"], -4.0)

    def test_training_logs_epoch_and_selected_metrics_when_enabled(self):
        training = [record(), record("u", answer="")]
        tuning = [dict(record("t"), title="Tune"), dict(record("tu", answer=""), title="Tune")]
        manifest = {"titles": {"train": ["Train"], "tuning": ["Tune"], "validation": ["Dev"]}}
        tracker = MagicMock(run_id="fixture-run")
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                limit=None, lr=2e-4, micro_batch=2, accum=2, epochs=1,
                batch_size=2, output_dir=directory, train_path=run_roberta.TRAIN_PATH,
                max_epochs=None, patience=3, min_delta=0.1, log_every=50,
                mlflow=True, mlflow_log_adapter=False,
                training_mode="lora",
            )
            fixtures = {args.train_path: training, run_roberta.TUNING_PATH: tuning,
                        run_roberta.MANIFEST_PATH: manifest}
            with patch.object(run_roberta, "optional_run",
                              return_value=contextlib.nullcontext(tracker)), \
                    patch.object(run_roberta, "load_json", side_effect=fixtures.__getitem__), \
                    patch.object(run_roberta, "file_sha256", return_value="fixture-hash"), \
                    patch.object(run_roberta, "retrieval_contexts",
                                 return_value=([[r["context"]] for r in tuning], {})), \
                    patch.object(run_roberta.AutoTokenizer, "from_pretrained",
                                 return_value=self.tokenizer), \
                    patch.object(run_roberta, "load_reader",
                                 return_value=add_lora(tiny_model(len(self.tokenizer)))), \
                    contextlib.redirect_stdout(io.StringIO()):
                run_roberta.train(args, "cpu")
            with open(os.path.join(directory, "selection.json"), encoding="utf-8") as f:
                self.assertEqual(json.load(f)["mlflow_run_id"], "fixture-run")
            self.assertTrue(any("tuning.overall_f1" in call.args[0]
                                and call.kwargs.get("step") == 1
                                for call in tracker.log_metrics.call_args_list))
            self.assertTrue(any("selected.tuning.overall_f1" in call.args[0]
                                for call in tracker.log_metrics.call_args_list))
            tracker.log_params.assert_called_once()

    def test_head_only_freezes_encoder_and_round_trips_compact_checkpoint(self):
        torch.manual_seed(42)
        base = tiny_model(len(self.tokenizer))
        encoder_state = copy.deepcopy(base.roberta.state_dict())
        original_head = base.qa_outputs.weight.detach().clone()
        with patch("squad_qa.roberta_qa.AutoModelForQuestionAnswering.from_pretrained",
                   return_value=base):
            model = load_reader(train=True, training_mode="head_only")
        self.assertEqual([name for name, p in model.named_parameters() if p.requires_grad],
                         ["qa_outputs.weight", "qa_outputs.bias"])
        model.train()
        model.roberta.eval()
        r = record()
        features = make_features(self.tokenizer, [r], [[r["context"]]])
        batch = collate_features(features, self.tokenizer.pad_token_id, "cpu", labels=True)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
        model(**batch).loss.backward()
        self.assertTrue(all(p.grad is None for p in model.roberta.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(original_head, model.qa_outputs.weight))
        for name, value in model.roberta.state_dict().items():
            torch.testing.assert_close(value, encoder_state[name])
        model.eval()
        expected = model(**batch).start_logits.detach()
        with tempfile.TemporaryDirectory() as path:
            save_reader(model, path, "head_only")
            self.assertEqual(os.listdir(path), ["qa_head.safetensors"])
            restored_base = tiny_model(len(self.tokenizer))
            restored_base.roberta.load_state_dict(encoder_state)
            with patch("squad_qa.roberta_qa.AutoModelForQuestionAnswering.from_pretrained",
                       return_value=restored_base):
                restored = load_reader(adapter=path, training_mode="head_only").eval()
            self.assertTrue(all(not p.requires_grad for p in restored.parameters()))
            torch.testing.assert_close(restored(**batch).start_logits, expected)

    def test_head_only_training_keeps_encoder_eval_and_records_mode(self):
        training = [record(), record("u", answer="")]
        tuning = [dict(record("t"), title="Tune"), dict(record("tu", answer=""), title="Tune")]
        manifest = {"titles": {"train": ["Train"], "tuning": ["Tune"], "validation": ["Dev"]}}
        base = tiny_model(len(self.tokenizer))
        encoder_state = copy.deepcopy(base.roberta.state_dict())
        encoder_training_states = []
        hook = base.roberta.register_forward_pre_hook(
            lambda module, inputs: encoder_training_states.append(module.training)
        )
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                limit=None, lr=2e-4, micro_batch=2, accum=2, epochs=1,
                batch_size=2, output_dir=directory, train_path=run_roberta.TRAIN_PATH,
                max_epochs=None, patience=3, min_delta=0.1, log_every=50,
                mlflow=False, training_mode="head_only",
            )
            fixtures = {args.train_path: training, run_roberta.TUNING_PATH: tuning,
                        run_roberta.MANIFEST_PATH: manifest}
            with patch.object(run_roberta, "load_json", side_effect=fixtures.__getitem__), \
                    patch.object(run_roberta, "file_sha256", return_value="fixture-hash"), \
                    patch.object(run_roberta, "retrieval_contexts",
                                 return_value=([[r["context"]] for r in tuning], {})), \
                    patch.object(run_roberta.AutoTokenizer, "from_pretrained",
                                 return_value=self.tokenizer), \
                    patch("squad_qa.roberta_qa.AutoModelForQuestionAnswering.from_pretrained",
                          return_value=base), contextlib.redirect_stdout(io.StringIO()):
                run_roberta.train(args, "cpu")
            hook.remove()
            self.assertTrue(encoder_training_states)
            self.assertFalse(any(encoder_training_states))
            with open(os.path.join(directory, "selection.json"), encoding="utf-8") as f:
                selection = json.load(f)
            self.assertEqual(selection["training_mode"], "head_only")
            self.assertFalse(selection["lora_enabled"])
            self.assertEqual(selection["lora"], {})
            self.assertEqual(selection["encoder_mode"], "eval")
            self.assertEqual(selection["trainable_parameters"], 50)
            for name, value in base.roberta.state_dict().items():
                torch.testing.assert_close(value, encoder_state[name])
            adapter = selection["selected_adapter"]
            self.assertTrue(os.path.isfile(os.path.join(adapter, "qa_head.safetensors")))
            self.assertFalse(os.path.exists(os.path.join(adapter, "adapter_model.safetensors")))
            with open(os.path.join(adapter, "qa_config.json"), encoding="utf-8") as f:
                self.assertEqual(json.load(f)["training_mode"], "head_only")

    def test_calibrated_loader_supports_head_only_and_legacy_lora(self):
        config = {
            "base_model": run_roberta.MODEL_NAME, "max_length": 384, "stride": 128,
            "max_answer_length": 30, "top_k": 3, "context": "retrieved top-3", "threshold": 0.0,
        }
        for mode in ("lora", "head_only"):
            metadata = dict(config)
            if mode == "head_only":
                metadata["training_mode"] = mode
            with self.subTest(mode=mode), \
                    patch.object(run_roberta, "load_json", return_value=metadata), \
                    patch.object(run_roberta.AutoTokenizer, "from_pretrained"), \
                    patch.object(run_roberta, "load_reader") as loader:
                run_roberta.load_calibrated_reader("checkpoint", "cpu")
                loader.assert_called_once_with(
                    adapter="checkpoint", device="cpu", training_mode=mode
                )

    def test_unknown_training_mode_fails_before_loading_model(self):
        with patch("squad_qa.roberta_qa.AutoModelForQuestionAnswering.from_pretrained") as loader:
            with self.assertRaisesRegex(ValueError, "Unknown reader training mode"):
                load_reader(train=True, training_mode="unknown")
            loader.assert_not_called()

    def test_manifest_leakage_rejected(self):
        manifest = {"titles": {"train": ["Train"], "tuning": ["Train"], "validation": []}}
        with self.assertRaisesRegex(ValueError, "overlapping titles"):
            run_roberta.check_splits([record()], [record("other")], manifest)

    def test_evaluate_writes_squad_maps_without_retuning(self):
        records = [record(), record("u", answer="")]
        model = add_lora(tiny_model(len(self.tokenizer)))
        config = {"threshold": 0.0}
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(adapter=directory, batch_size=2, mlflow=False)
            with patch.object(run_roberta, "load_json", return_value=records), \
                    patch.object(run_roberta, "retrieval_contexts",
                                 return_value=([[r["context"]] for r in records], {})), \
                    patch.object(run_roberta, "load_calibrated_reader",
                                 return_value=(model, self.tokenizer, config)), \
                    patch.object(run_roberta, "tune_threshold",
                                 side_effect=AssertionError("Dev must not retune")), \
                    contextlib.redirect_stdout(io.StringIO()):
                run_roberta.evaluate(args, "cpu")
            with open(os.path.join(directory, "dev_predictions.json"), encoding="utf-8") as f:
                predictions = json.load(f)
            self.assertEqual(set(predictions), {"oracle", "retrieved"})
            self.assertEqual(set(predictions["retrieved"]), {"a", "u"})
            self.assertTrue(all(isinstance(value, str)
                                for value in predictions["retrieved"].values()))
            with open(os.path.join(directory, "dev_results.json"), encoding="utf-8") as f:
                results = json.load(f)
            self.assertEqual(results["config"]["threshold"], 0.0)
            self.assertEqual(results["retrieved"]["all"]["total"], 2)

class EarlyStoppingTests(unittest.TestCase):
    def test_no_stop_before_initial_budget_even_with_long_plateau(self):
        stopper = run_roberta.EarlyStopping(5, 3, 0.1)
        for epoch in range(1, 5):
            self.assertFalse(stopper.update(epoch, 50.0))
        self.assertTrue(stopper.update(5, 50.0))

    def test_significant_improvements_reset_patience(self):
        stopper = run_roberta.EarlyStopping(5, 3, 0.1)
        scores = [50.0, 50.05, 50.1, 50.15, 50.2, 50.2, 50.2]
        for epoch, score in enumerate(scores, 1):
            self.assertFalse(stopper.update(epoch, score))
        self.assertTrue(stopper.update(8, 50.2))

    def test_zero_delta_ties_do_not_count_as_improvements(self):
        stopper = run_roberta.EarlyStopping(1, 2, 0.0)
        self.assertFalse(stopper.update(1, 50.0))
        self.assertFalse(stopper.update(2, 50.0))
        self.assertTrue(stopper.update(3, 50.0))

    def test_nonfinite_metric_rejected(self):
        for value in (float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "finite tuning F1"):
                run_roberta.EarlyStopping(5, 3, 0.1).update(1, value)


if __name__ == "__main__":
    unittest.main()
