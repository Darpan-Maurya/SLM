"""Evaluation."""
from slm.eval.harness import (
    Doc,
    Request,
    Score,
    evaluate_docs,
    format_table,
    score_requests,
    token_perplexity,
)
from slm.eval.tasks import DEFAULT_SUITE, REGISTRY, load_task

__all__ = [
    "DEFAULT_SUITE",
    "REGISTRY",
    "Doc",
    "Request",
    "Score",
    "evaluate_docs",
    "format_table",
    "load_task",
    "score_requests",
    "token_perplexity",
]
