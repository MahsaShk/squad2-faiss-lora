"""Offline tests for shared retrieval and the standalone training index builder."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import faiss
import numpy as np

from squad_qa import build_training_index, retrieval


class RetrievalTests(unittest.TestCase):
    def test_chunks_preserve_overlap_and_short_text(self):
        text = " ".join(f"{i:04d}" for i in range(600))
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "offset_mapping": [(i * 5, i * 5 + 4) for i in range(600)],
        }
        chunks = retrieval.chunk_text(tokenizer, text, 254)
        self.assertEqual(len(chunks), 4)
        self.assertEqual([chunk.split()[0] for chunk in chunks],
                         ["0000", "0127", "0254", "0381"])
        self.assertEqual([len(chunk.split()) for chunk in chunks], [254, 254, 254, 219])
        self.assertEqual(chunks[0].split()[127:], chunks[1].split()[:127])
        tokenizer.return_value = {"offset_mapping": [(0, 4)]}
        self.assertEqual(retrieval.chunk_text(tokenizer, "word", 254), ["word"])

    def test_deduplication_preserves_rank_and_ignores_negative_hits(self):
        self.assertEqual(retrieval.dedupe_owners([-1, 2, 1, 0, 2], [4, 4, 9]), [9, 4])

    def test_search_caps_hits_and_uses_normalized_float32_vectors(self):
        model, index = MagicMock(), MagicMock()
        model.encode.return_value = np.array([[1.0, 0.0]], dtype="float64")
        index.ntotal = 3
        index.search.return_value = (None, [[2, 0, 1]])
        result = retrieval.search(model, index, [0, 0, 1], [{"question": "Why?"}])
        self.assertEqual(result, [[1, 0]])
        vectors, k = index.search.call_args.args
        self.assertEqual(vectors.dtype, np.dtype("float32"))
        self.assertEqual(k, 3)
        model.encode.assert_called_once_with(
            ["Why?"], normalize_embeddings=True, convert_to_numpy=True
        )

    def test_search_rejects_empty_index_or_mismatched_owners(self):
        model, index = MagicMock(), MagicMock()
        index.ntotal = 0
        with self.assertRaisesRegex(ValueError, "empty FAISS"):
            retrieval.search(model, index, [], [{"question": "Why?"}])
        index.ntotal = 2
        with self.assertRaisesRegex(ValueError, "owners"):
            retrieval.search(model, index, [0], [{"question": "Why?"}])
        model.encode.assert_not_called()

    def test_contexts_deduplicate_and_ignore_negative_faiss_hits(self):
        index = MagicMock()
        index.ntotal = 5
        index.search.return_value = (None, [[0, 1, -1, 2, 3]])
        fixtures = {
            retrieval.PASSAGES_PATH: ["first", "second", "third"],
            retrieval.CHUNKED_OWNERS_PATH: [0, 0, 1, 2, 2],
        }
        with patch.object(retrieval, "load_json", side_effect=fixtures.__getitem__), \
                patch.object(retrieval.faiss, "read_index", return_value=index), \
                patch.object(retrieval, "SentenceTransformer") as embedder:
            embedder.return_value.encode.return_value = [[1.0, 0.0]]
            contexts, recall = retrieval.retrieval_contexts(
                [{"question": "Why?", "context": "second", "answers": {"text": []}}],
                "dev", "cpu",
            )
        self.assertEqual(contexts, [["first", "second", "third"]])
        self.assertEqual(recall["recall@3_all"], 1.0)
        self.assertIsNone(recall["recall@3_hasans"])

    def test_contexts_reject_invalid_split_and_tuning_title_leakage(self):
        with self.assertRaisesRegex(ValueError, "split"):
            retrieval.retrieval_contexts([], "unknown", "cpu")
        fixtures = {
            retrieval.TRAIN_PARAGRAPHS_PATH: [{"title": "Dev", "context": "text"}],
            retrieval.MANIFEST_PATH: {"titles": {"train": ["Train"], "tuning": ["Tune"]}},
        }
        with patch.object(retrieval, "load_json", side_effect=fixtures.__getitem__), \
                self.assertRaisesRegex(ValueError, "outside the split manifest"):
            retrieval.retrieval_contexts([], "tuning", "cpu")


class TrainingIndexTests(unittest.TestCase):
    def test_collects_only_assigned_titles_and_unique_contexts(self):
        source = {
            "title": ["Train", "Tune", "Dev", "Train"],
            "context": ["first", "second", "excluded", "first"],
        }
        paragraphs, ids = build_training_index.collect_paragraphs(source, {"Train", "Tune"})
        self.assertEqual(paragraphs, [
            {"title": "Train", "context": "first"}, {"title": "Tune", "context": "second"},
        ])
        self.assertEqual(ids, {"first": 0, "second": 1})

    def test_build_writes_faiss_and_matching_parent_metadata(self):
        paragraphs = [{"title": "Train", "context": "first"},
                      {"title": "Tune", "context": "second"}]
        model = MagicMock()
        model.max_seq_length = 256
        model.tokenizer.return_value = {"offset_mapping": [(0, 3)]}
        model.get_sentence_embedding_dimension.return_value = 2
        model.encode.return_value = np.array([[1.0, 0.0], [0.0, 1.0]])
        with tempfile.TemporaryDirectory() as directory:
            paths = {
                "INDEX_PATH": os.path.join(directory, "index.faiss"),
                "OWNERS_PATH": os.path.join(directory, "owners.json"),
                "PARAGRAPHS_PATH": os.path.join(directory, "paragraphs.json"),
            }
            with patch.multiple(build_training_index, **paths), \
                    contextlib.redirect_stdout(io.StringIO()):
                index, owners = build_training_index.build_index(model, paragraphs)
            self.assertEqual(index.ntotal, 2)
            self.assertEqual(faiss.read_index(paths["INDEX_PATH"]).ntotal, 2)
            self.assertEqual(owners, [0, 1])
            with open(paths["OWNERS_PATH"], encoding="utf-8") as f:
                self.assertEqual(json.load(f), owners)
            with open(paths["PARAGRAPHS_PATH"], encoding="utf-8") as f:
                self.assertEqual(json.load(f), paragraphs)

    def test_builder_rejects_title_overlap_before_loading_source_dataset(self):
        fixtures = {
            build_training_index.MANIFEST_PATH: {"titles": {
                "train": ["Train"], "tuning": ["Train"], "validation": ["Dev"],
            }},
            build_training_index.TRAIN_PATH: [],
            build_training_index.TUNING_PATH: [],
        }
        with patch.object(build_training_index, "load_json", side_effect=fixtures.__getitem__), \
                patch.object(build_training_index, "load_dataset") as load, \
                self.assertRaisesRegex(ValueError, "Title leakage"):
            build_training_index.main("cpu")
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
