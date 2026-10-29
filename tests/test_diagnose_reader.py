"""Offline tests for fixed-threshold extraction/abstention diagnostics."""
import argparse
import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from squad_qa import diagnose_reader


def record(qid, answers):
    context = "North and West Africa"
    return {
        "id": qid, "title": "Tune", "context": context, "question": "Where?",
        "answers": {"text": answers, "answer_start": [context.index(a) for a in answers]},
    }


def candidate(text, margin):
    return {"answer": text, "na_score": margin, "passage": text,
            "start_char": 0, "end_char": len(text)}


class AnalysisTests(unittest.TestCase):
    def test_buckets_multiple_references_partial_credit_and_threshold_equality(self):
        records = [
            record("a", ["Africa", "North and West Africa"]),
            record("b", ["Africa"]), record("c", ["North and West Africa"]),
            record("d", ["Africa"]), record("u", []),
        ]
        rows = [candidate("NORTH AND WEST AFRICA.", 0.0), candidate("Africa", 1.0),
                candidate("West Africa", -1.0), candidate("Europe", 2.0),
                candidate("Africa", 1.0)]
        summary, examples = diagnose_reader.analyze_predictions(
            records, rows, 0.0, [True, True, False, False, True], "retrieved"
        )
        answerable = summary["answerable"]
        self.assertEqual(answerable["buckets"], {
            "correct_answered": 1, "correct_abstained": 1,
            "incorrect_answered": 1, "incorrect_abstained": 1,
        })
        self.assertEqual(answerable["questions"], 4)
        self.assertEqual(answerable["before_abstention"]["exact"], 50)
        self.assertEqual(answerable["after_abstention"]["exact"], 25)
        self.assertAlmostEqual(answerable["before_abstention"]["f1"], 100 * (2 + 2 / 3) / 4)
        self.assertAlmostEqual(answerable["after_abstention"]["f1"], 100 * (1 + 2 / 3) / 4)
        self.assertEqual(answerable["partial_overlap_candidates"], 1)
        self.assertEqual(answerable["zero_overlap_candidates"], 1)
        self.assertEqual(summary["unanswerable"]["abstained"], 1)
        self.assertEqual(summary["answerable_by_gold_retrieved"]["present"]["questions"], 2)
        self.assertEqual(summary["answerable_by_gold_retrieved"]["missing"]["questions"], 2)
        self.assertFalse(examples[0]["abstained"])
        self.assertEqual(examples[0]["raw_exact"], 1)
        self.assertEqual(examples[1]["final_prediction"], "")
        self.assertFalse(examples[2]["gold_context_in_input"])

    def test_oracle_retains_retrieval_flag_but_always_has_gold_context(self):
        summary, rows = diagnose_reader.analyze_predictions(
            [record("a", ["Africa"])], [candidate("Africa", 0)], 0, [False], "oracle"
        )
        self.assertFalse(rows[0]["gold_retrieved_top3"])
        self.assertTrue(rows[0]["gold_context_in_input"])
        self.assertIsNone(summary["answerable_by_gold_retrieved"]["present"]["before_abstention"]["f1"])

    def test_invalid_candidates_and_lengths_fail_explicitly(self):
        records = [record("a", ["Africa"])]
        with self.assertRaisesRegex(ValueError, "retrieval flag"):
            diagnose_reader.analyze_predictions(records, [], 0, [True], "oracle")
        for value in (float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "finite"):
                diagnose_reader.analyze_predictions(
                    records, [candidate("Africa", 0)], value, [True], "oracle"
                )
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                diagnose_reader.analyze_predictions(
                    records, [candidate("Africa", value)], 0, [True], "oracle"
                )
        bad = candidate("Africa", 0)
        bad["end_char"] = 1
        with self.assertRaisesRegex(ValueError, "source span"):
            diagnose_reader.analyze_predictions(records, [bad], 0, [True], "oracle")

    def test_tuning_only_run_preserves_threshold_weights_and_accounting(self):
        records = [record("t", ["Africa"]), record("u", [])]
        rows = [candidate("Africa", 1.0), candidate("Europe", 2.0)]
        model = MagicMock()
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                adapter=directory, output_dir=None, batch_size=2, device="cpu"
            )
            fixtures = {
                diagnose_reader.TUNING_PATH: records,
                os.path.join(directory, "qa_config.json"): {"threshold": 0.0},
            }
            with patch.object(diagnose_reader, "file_sha256", return_value="unchanged"), \
                    patch.object(diagnose_reader, "load_json",
                                 side_effect=fixtures.__getitem__) as load, \
                    patch.object(diagnose_reader, "retrieval_contexts", return_value=(
                        [[r["context"]] for r in records], {"recall@3_all": 1.0}
                    )) as retrieve, \
                    patch.object(diagnose_reader, "load_calibrated_reader",
                                 return_value=(model, object(), {"threshold": 0.0})), \
                    patch.object(diagnose_reader, "predict", return_value=rows) as predict, \
                    contextlib.redirect_stdout(io.StringIO()):
                diagnose_reader.run(args)
            self.assertEqual({call.args[0] for call in load.call_args_list}, set(fixtures))
            retrieve.assert_called_once_with(records, "tuning", "cpu")
            model.requires_grad_.assert_called_once_with(False)
            model.train.assert_not_called()
            self.assertEqual(predict.call_count, 2)
            with open(os.path.join(directory, "tuning_diagnostics", "summary.json"),
                      encoding="utf-8") as f:
                report = json.load(f)
            self.assertEqual(report["threshold"], 0)
            self.assertFalse(report["threshold_retuned"])
            self.assertFalse(report["weights_updated"])
            for condition in ("oracle", "retrieved"):
                summary = report["conditions"][condition]
                self.assertEqual(summary["answerable"]["buckets"]["correct_abstained"], 1)
                self.assertEqual(summary["unanswerable"]["abstained"], 1)
            self.assertTrue(os.path.isfile(os.path.join(
                directory, "tuning_diagnostics", "examples.json"
            )))


if __name__ == "__main__":
    unittest.main()
