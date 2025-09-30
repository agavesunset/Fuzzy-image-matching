"""Fuzzy image matching package."""

from .matching import (
    FuzzyImageMatcher,
    ImageProcessingError,
    MatchResult,
    compute_histogram_similarity,
    compute_orb_similarity,
    compute_ssim_similarity,
)

__all__ = [
    "FuzzyImageMatcher",
    "ImageProcessingError",
    "MatchResult",
    "compute_histogram_similarity",
    "compute_orb_similarity",
    "compute_ssim_similarity",
]
