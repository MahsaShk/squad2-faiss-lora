"""Tests for reproducible LoRA data splits and leakage checks."""
import contextlib
import copy
import io
import json
import os
import random
import tempfile
import unittest
from unittest.mock import patch

from squad_qa import prepare_lora_data
from squad_qa.prepare_lora_data import expand_training, file_sha256, make_splits, sample_balanced


def examples():
    return [
        {
            "id": f"{title}-{index}",
            "title": title,
            "context": "example passage",
            "question": f"question {index}",
            "answers": {
                "text": ["example"] if index % 2 else [],
                "answer_start": [0] if index % 2 else [],
            },
        }
        for title in ("A", "B", "C", "D")
        for index in range(8)
    ]


class SplitTests(unittest.TestCase):
    def split(self, records=None, validation_titles=("held-out",), seed=42):
        return make_splits(
            examples() if records is None else records,
            validation_titles,
            seed=seed,
            tuning_title_count=1,
            train_size=8,
            tuning_size=4,
        )

    def test_sizes_balance_schema_and_disjointness(self):
        train, tuning, manifest = self.split()
        source = {record["id"]: record for record in examples()}
        for subset, expected_size in ((train, 8), (tuning, 4)):
            self.assertEqual(len(subset), expected_size)
            self.assertEqual(
                sum(bool(record["answers"]["text"]) for record in subset),
                expected_size // 2,
            )
            for record in subset:
                self.assertEqual(record, source[record["id"]])
        self.assertFalse({r["id"] for r in train} & {r["id"] for r in tuning})
        self.assertFalse({r["title"] for r in train} & {r["title"] for r in tuning})
        self.assertEqual(len(manifest["titles"]["train"]), 3)
        self.assertEqual(len(manifest["titles"]["tuning"]), 1)
        self.assertEqual(manifest["counts"]["train"]["questions"], 8)
        self.assertEqual(
            set(manifest["titles"]["train"]) | set(manifest["titles"]["tuning"]),
            {"A", "B", "C", "D"},
        )

    def test_reproducible_without_changing_global_rng(self):
        state = random.getstate()
        self.assertEqual(self.split(), self.split())
        self.assertEqual(random.getstate(), state)
        self.assertNotEqual(self.split(), self.split(seed=43))

    def test_validation_title_leakage_rejected_before_sampling(self):
        with self.assertRaisesRegex(ValueError, "Title leakage"):
            self.split(validation_titles=("A",))

    def test_duplicate_ids_rejected(self):
        records = examples()
        with self.assertRaisesRegex(ValueError, "duplicate question IDs"):
            self.split(records=records + [records[0]])

    def test_invalid_title_counts_rejected(self):
        for count in (0, 4, 5):
            with self.subTest(count=count):
                with self.assertRaisesRegex(ValueError, "Tuning title count"):
                    make_splits(examples(), {"held-out"}, tuning_title_count=count)

    def test_invalid_sizes_and_insufficient_groups_rejected(self):
        for size in (0, -2, 3):
            with self.subTest(size=size):
                with self.assertRaisesRegex(ValueError, "positive even"):
                    sample_balanced(examples(), size, random.Random(42))
        with self.assertRaisesRegex(ValueError, "questions per answerability group"):
            sample_balanced(examples(), 34, random.Random(42))
        answerable_only = [r for r in examples() if r["answers"]["text"]]
        with self.assertRaisesRegex(ValueError, "questions per answerability group"):
            sample_balanced(answerable_only, 2, random.Random(42))

    def expansion_fixture(self):
        train, tuning, manifest = self.split()
        dev = [dict(examples()[0], id="dev-only", title="held-out")]
        return train, tuning, dev, manifest

    def test_expansion_preserves_existing_examples_titles_and_inputs(self):
        train, tuning, dev, manifest = self.expansion_fixture()
        frozen = copy.deepcopy((train, tuning, dev, manifest))
        expanded, expanded_manifest = expand_training(
            examples(), train, tuning, dev, manifest, 20
        )
        self.assertEqual(len(expanded), 20)
        self.assertEqual(len({r["id"] for r in expanded}), 20)
        self.assertEqual(sum(bool(r["answers"]["text"]) for r in expanded), 10)
        self.assertTrue({r["id"] for r in train} <= {r["id"] for r in expanded})
        self.assertTrue({r["title"] for r in expanded} <= set(manifest["titles"]["train"]))
        self.assertEqual(expanded_manifest["titles"], manifest["titles"])
        self.assertEqual(expanded_manifest["counts"]["tuning"], manifest["counts"]["tuning"])
        self.assertEqual((train, tuning, dev, manifest), frozen)
        self.assertEqual(
            (expanded, expanded_manifest),
            expand_training(examples(), train, tuning, dev, manifest, 20),
        )

    def test_expansion_rejects_leakage_and_source_changes(self):
        train, tuning, dev, manifest = self.expansion_fixture()
        overlapping = copy.deepcopy(manifest)
        overlapping["titles"]["tuning"].append(manifest["titles"]["train"][0])
        with self.assertRaisesRegex(ValueError, "Title leakage"):
            expand_training(examples(), train, tuning, dev, overlapping, 20)
        changed = copy.deepcopy(train)
        changed[0]["context"] = "changed"
        with self.assertRaisesRegex(ValueError, "differs from source"):
            expand_training(examples(), changed, tuning, dev, manifest, 20)
        duplicate = copy.deepcopy(dev)
        duplicate[0]["id"] = train[0]["id"]
        with self.assertRaisesRegex(ValueError, "overlapping question IDs"):
            expand_training(examples(), train, tuning, duplicate, manifest, 20)
        with self.assertRaisesRegex(ValueError, "duplicate question IDs"):
            expand_training(examples() + [examples()[0]], train, tuning, dev, manifest, 20)
        for size in (0, 8, 19, 40):
            with self.subTest(size=size), self.assertRaises(ValueError):
                expand_training(examples(), train, tuning, dev, manifest, size)

    def test_expansion_cli_preserves_frozen_files_byte_for_byte(self):
        train, tuning, dev, manifest = self.expansion_fixture()
        with tempfile.TemporaryDirectory() as directory:
            paths = {name: os.path.join(directory, name + ".json") for name in
                     ("TRAIN_PATH", "TUNING_PATH", "DEV_PATH", "MANIFEST_PATH")}
            for name, value in zip(paths, (train, tuning, dev, manifest)):
                with open(paths[name], "w", encoding="utf-8") as f:
                    json.dump(value, f)
            before = {p: file_sha256(p) for p in paths.values()}
            output = os.path.join(directory, "train_20.json")
            with patch.multiple(prepare_lora_data, **paths), \
                    patch.object(prepare_lora_data, "load_dataset", return_value={
                        "train": examples(), "validation": {"title": ["held-out"]},
                    }), contextlib.redirect_stdout(io.StringIO()):
                prepare_lora_data.main(20, output)
                with self.assertRaises(FileExistsError):
                    prepare_lora_data.main(20, output)
            self.assertEqual(before, {p: file_sha256(p) for p in paths.values()})
            with open(output, encoding="utf-8") as f:
                self.assertEqual(len(json.load(f)), 20)
            with open(os.path.splitext(output)[0] + "_manifest.json", encoding="utf-8") as f:
                expanded_manifest = json.load(f)
            self.assertEqual(expanded_manifest["frozen_input_sha256"], before)
            self.assertEqual(expanded_manifest["counts"]["train"]["answerable"], 10)


if __name__ == "__main__":
    unittest.main()
