"""Unit tests for the eval metrics — no LLM, fully deterministic."""

from ctmatch.eval.metrics import accuracy, confident_accuracy, unsafe_overclaim_rate


def test_accuracy_basic():
    pairs = [("met", "met"), ("unknown", "not_met"), ("unknown", "unknown")]
    assert abs(accuracy(pairs) - 2 / 3) < 1e-9


def test_confident_accuracy_ignores_unknown_gold():
    pairs = [("met", "met"), ("met", "not_met"), ("met", "unknown")]
    assert confident_accuracy(pairs) == 0.5


def test_unsafe_overclaim_rate():
    pairs = [("met", "unknown"), ("not_met", "unknown"), ("unknown", "unknown")]
    assert abs(unsafe_overclaim_rate(pairs) - 2 / 3) < 1e-9


def test_no_unknowns_means_zero_overclaim():
    assert unsafe_overclaim_rate([("met", "met"), ("not_met", "not_met")]) == 0.0
