"""Unit tests that run with no network, models, or Qdrant — so CI is fast and
deterministic. The fusion logic is the most error-prone pure function."""

from ctmatch.retrieval import reciprocal_rank_fusion


def test_rrf_rewards_agreement_across_lists():
    dense = ["a", "b", "c"]
    sparse = ["b", "d", "a"]
    fused = reciprocal_rank_fusion([dense, sparse])
    assert fused[0] == "b"
    assert set(fused) == {"a", "b", "c", "d"}


def test_rrf_handles_single_list():
    assert reciprocal_rank_fusion([["x", "y", "z"]]) == ["x", "y", "z"]


def test_rrf_empty():
    assert reciprocal_rank_fusion([]) == []
