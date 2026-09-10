"""Deterministic policy that follows the first presented candidate."""

from __future__ import annotations

from src.rag.agent.schema import Decision


def decide(question: str, observation: str, candidates: list[int]) -> Decision:
    """Return a follow decision for the first candidate without an LLM call."""
    if not isinstance(question, str) or not isinstance(observation, str):
        raise ValueError("question and observation must be strings")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates must be a non-empty list")
    if any(isinstance(candidate, bool) or not isinstance(candidate, int) or candidate <= 0 for candidate in candidates):
        raise ValueError("candidates must contain positive integers")
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidates must not contain duplicates")
    return Decision(stop=False, dieu=candidates[0])
