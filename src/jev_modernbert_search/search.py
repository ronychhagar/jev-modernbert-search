from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

DEFAULT_MODEL = "Alibaba-NLP/gte-modernbert-base"


@dataclass(frozen=True)
class SearchResult:
    text: str
    embedding_score: float
    jev_score: float | None = None


class ModernBertSearch:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Install project dependencies before starting search."
            ) from exc

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        return np.asarray(
            self.model.encode(
                texts,
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=True,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )

    def retrieve(
        self,
        query: str,
        documents: list[str],
        document_embeddings: np.ndarray,
        limit: int = 20,
    ) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("Search query must not be empty.")
        if len(documents) != len(document_embeddings):
            raise ValueError("Each document must have exactly one embedding.")
        if limit < 1:
            raise ValueError("limit must be at least 1.")
        if not documents:
            return []

        query_embedding = self.encode([query])[0]
        scores = document_embeddings @ query_embedding
        query_key = query.casefold().strip()
        ranked = np.argsort(-scores)
        results: list[SearchResult] = []
        seen: set[str] = set()
        for index in ranked:
            text = documents[int(index)]
            key = text.casefold().strip()
            if key == query_key or key in seen:
                continue
            seen.add(key)
            results.append(SearchResult(text=text, embedding_score=float(scores[index])))
            if len(results) == limit:
                break
        return results


async def rerank_with_jev(
    query: str,
    results: list[SearchResult],
    contexts: list[str] | None = None,
) -> list[SearchResult]:
    if not results:
        return []
    if contexts is not None and len(contexts) != len(results):
        raise ValueError("Each search result must have exactly one context.")

    api_key = os.environ.get("JEV_API_KEY")
    if not api_key:
        raise RuntimeError("Set JEV_API_KEY to use TypeSafe Jev reranking.")

    try:
        from jevclient import JevClient, Noul
    except ImportError as exc:
        raise RuntimeError("Install project dependencies before using Jev reranking.") from exc

    state: dict[str, object] = {"query": query}
    questions = {}
    for index, result in enumerate(results):
        result_key = f"result_{index}"
        state[result_key] = result.text
        if contexts is not None and contexts[index]:
            context_key = f"context_{index}"
            state[context_key] = contexts[index]
            context_instruction = f" Use ontology context in `{context_key}`."
        else:
            context_instruction = ""
        questions[f"relevant_{index}"] = Noul(
            f"Is `{result_key}` a relevant search result for the intent in `query`?"
            f"{context_instruction}"
        )

    async with JevClient(api_key) as jev:
        answers = await jev.ask(state, questions)

    ranked = [
        SearchResult(
            text=result.text,
            embedding_score=result.embedding_score,
            jev_score=float(answers[f"relevant_{index}"].noul),
        )
        for index, result in enumerate(results)
    ]
    return sorted(
        ranked,
        key=lambda result: (
            result.jev_score if result.jev_score is not None else 0.0,
            result.embedding_score,
        ),
        reverse=True,
    )
