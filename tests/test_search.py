import tempfile
import unittest
from pathlib import Path

import numpy as np

from jev_modernbert_search.index import load_or_build_index
from jev_modernbert_search.search import ModernBertSearch


class FakeModel:
    def __init__(self):
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        return np.full((len(texts), 2), float(len(texts)), dtype=np.float32)


class SearchTests(unittest.TestCase):
    def test_retrieve_excludes_query_and_ranks_by_cosine_similarity(self):
        search = ModernBertSearch.__new__(ModernBertSearch)
        search.encode = lambda texts: np.array([[1.0, 0.0]], dtype=np.float32)

        results = search.retrieve(
            "query",
            ["query", "A close match", "Unrelated"],
            np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float32),
            limit=2,
        )

        self.assertEqual([result.text for result in results], ["A close match", "Unrelated"])
        self.assertAlmostEqual(results[0].embedding_score, 0.8)

    def test_index_cache_reuses_embeddings_and_invalidates_changed_documents(self):
        with tempfile.TemporaryDirectory() as directory:
            model = FakeModel()
            cache_path = Path(directory) / "index.npz"

            first = load_or_build_index(model, "fake-model", ["one"], cache_path)
            cached = load_or_build_index(model, "fake-model", ["one"], cache_path)
            changed = load_or_build_index(model, "fake-model", ["one", "two"], cache_path)

            self.assertEqual(model.calls, 2)
            np.testing.assert_array_equal(first, cached)
            self.assertEqual(changed.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
