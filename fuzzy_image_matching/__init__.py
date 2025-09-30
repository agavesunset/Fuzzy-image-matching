"""Fuzzy image matching package."""

from .matching import (
    FuzzyImageMatcher,
    MatchResult,
    compute_histogram_similarity,
    compute_orb_similarity,
    compute_ssim_similarity,
)

__all__ = [
    "FuzzyImageMatcher",
    "MatchResult",
    "compute_histogram_similarity",
    "compute_orb_similarity",
    "compute_ssim_similarity",
]
