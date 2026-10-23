from __future__ import annotations

from typing import Protocol, Sequence


class Reranker(Protocol):
    def rerank(self, candidates: Sequence[tuple[str, float]]) -> list[tuple[str, float]]: ...


class NoReranker:
    def rerank(self, candidates: Sequence[tuple[str, float]]) -> list[tuple[str, float]]:
        return list(candidates)

