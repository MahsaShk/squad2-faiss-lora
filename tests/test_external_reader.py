"""Offline tests for external-reader inference and tuning-only calibration."""
import argparse
import contextlib
import copy
import io
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from squad_qa import external_reader


def record(qid, title, answerable):
    return {
        "id": qid, "title": title, "context": "Paris is in France.",
        "question": "Where is Paris?",
        "answers": {"text": ["France"] if answerable else [],
                    "answer_start": [12] if answerable else []},
    }


def candidates(margins):
    return [{"answer": "France", "na_score": value, "start_char": 12, "end_char": 18,
             "passage": "Paris is in France."} for value in margins]


class ExternalReaderTests(unittest.TestCase):
    def test_inference_only_and_threshold_frozen_for_both_dev_conditions(self):
        tuning = [record("t", "Tune", True), record("tu", "Tune", False)]
        dev = [record("d", "Dev", True), record("du", "Dev", False)]
        fixtures = {external_reader.TUNING_PATH: tuning, external_reader.DEV_PATH: dev}
        model = MagicMock()
        model.to.return_value = model
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "baseline")
            args = argparse.Namespace(output_dir=output, device="cpu", batch_size=2, mlflow=False)
            with patch.object(external_reader, "load_json", side_effect=fixtures.__getitem__), \
                    patch.object(external_reader, "retrieval_contexts", side_effect=[
                        ([[r["context"]] for r in tuning], {"recall@3_all": 1.0}),
                        ([[r["context"]] for r in dev], {"recall@3_all": 1.0}),
                    ]), \
                    patch.object(external_reader, "model_info",
                                 return_value=argparse.Namespace(sha="test-revision")), \
                    patch.object(external_reader.AutoTokenizer, "from_pretrained") as tokenizer, \
                    patch.object(external_reader.AutoModelForQuestionAnswering, "from_pretrained",
                                 return_value=model) as loader, \
                    patch.object(external_reader, "predict", side_effect=[
                        candidates([-2.0, 1.0]), candidates([-1.0, 0.5]), candidates([-3.0, 1.0]),
                    ]) as predict, \
                    patch.object(external_reader, "tune_threshold",
                                 wraps=external_reader.tune_threshold) as tune, \
                    contextlib.redirect_stdout(io.StringIO()):
                external_reader.run(args)
            loader.assert_called_once_with(external_reader.MODEL_NAME, revision="test-revision")
            tokenizer.assert_called_once_with(
                external_reader.MODEL_NAME, revision="test-revision", use_fast=True
            )
            model.requires_grad_.assert_called_once_with(False)
            model.eval.assert_called_once()
            model.train.assert_not_called()
            self.assertEqual(predict.call_count, 3)
            tune.assert_called_once()
            self.assertEqual(tune.call_args.args[0], tuning)
            with open(os.path.join(output, "results.json"), encoding="utf-8") as f:
                results = json.load(f)
            self.assertEqual(results["config"]["model_revision"], "test-revision")
            self.assertFalse(results["config"]["trained_in_this_run"])
            self.assertFalse(results["config"]["lora"])
            self.assertEqual(results["calibration"]["threshold"], -2.0)
            self.assertEqual(results["dev"]["calibrated"]["threshold"], -2.0)
            self.assertEqual(results["dev"]["zero_margin"]["retrieved"]["all"]["f1"], 100)
            self.assertEqual(results["dev"]["calibrated"]["retrieved"]["all"]["f1"], 50)
            self.assertEqual(results["dev"]["calibrated"]["oracle"]["all"]["f1"], 100)
            with open(os.path.join(output, "predictions.json"), encoding="utf-8") as f:
                predictions = json.load(f)
            for mode in ("zero_margin", "calibrated"):
                for condition in ("oracle", "retrieved"):
                    self.assertEqual(set(predictions[mode][condition]), {"d", "du"})
                    self.assertTrue(all(isinstance(value, str)
                                        for value in predictions[mode][condition].values()))
            self.assertTrue(os.path.isfile(os.path.join(output, "calibration.json")))
            self.assertTrue(os.path.isfile(os.path.join(output, "candidates.json")))

    def test_existing_outputs_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(output_dir=directory, device="cpu", batch_size=2, mlflow=False)
            with self.assertRaises(FileExistsError):
                external_reader.run(args)

    def test_tuning_dev_overlap_rejected_before_model_loading(self):
        tuning = [record("same", "Tune", True)]
        for dev, message in (
            ([record("same", "Dev", True)], "IDs overlap"),
            ([record("other", "Tune", True)], "titles overlap"),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                args = argparse.Namespace(
                    output_dir=os.path.join(directory, "baseline"), device="cpu", batch_size=2,
                    mlflow=False,
                )
                fixtures = {external_reader.TUNING_PATH: copy.deepcopy(tuning),
                            external_reader.DEV_PATH: dev}
                with patch.object(external_reader, "load_json", side_effect=fixtures.__getitem__), \
                        patch.object(external_reader.AutoModelForQuestionAnswering,
                                     "from_pretrained") as loader, \
                        contextlib.redirect_stdout(io.StringIO()), \
                        self.assertRaisesRegex(ValueError, message):
                    external_reader.run(args)
                loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
