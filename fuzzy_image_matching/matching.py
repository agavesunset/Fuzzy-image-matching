"""Fuzzy image matching utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def _load_image(path: Path) -> np.ndarray:
    """Load an image from disk and ensure it is in BGR format."""
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    return image


def _ensure_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def compute_orb_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute similarity score based on ORB feature matching.

    Returns a score between 0 and 1.
    """
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    if descriptors1 is None or descriptors2 is None or len(keypoints1) == 0:
        return 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)
    if not matches:
        return 0.0

    distances = np.array([m.distance for m in matches], dtype=np.float32)
    score = np.exp(-distances.mean() / 50.0)
    return float(np.clip(score, 0.0, 1.0))


def compute_histogram_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute similarity using color histograms."""
    img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    hist_size = [8, 8, 8]
    ranges = [0, 180, 0, 256, 0, 256]
    channels = [0, 1, 2]

    hist1 = cv2.calcHist([img1_hsv], channels, None, hist_size, ranges)
    hist2 = cv2.calcHist([img2_hsv], channels, None, hist_size, ranges)

    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)

    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    score = (score + 1) / 2  # convert to 0-1
    return float(np.clip(score, 0.0, 1.0))


def compute_ssim_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Structural Similarity Index (SSIM)."""
    gray1 = _ensure_gray(img1)
    gray2 = _ensure_gray(img2)
    gray2_resized = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
    score, _ = structural_similarity(gray1, gray2_resized, full=True)
    return float(np.clip(score, 0.0, 1.0))


@dataclass
class MatchResult:
    candidate: Path
    score: float
    orb: float
    histogram: float
    ssim: float


class FuzzyImageMatcher:
    """Match a query image against a collection of candidate images."""

    def __init__(self, weights: Sequence[float] | None = None) -> None:
        if weights is None:
            weights = (0.4, 0.3, 0.3)
        if len(weights) != 3:
            raise ValueError("weights must contain exactly three values")
        self.weights = np.array(weights, dtype=np.float32)

    def score_pair(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[float, float, float, float]:
        orb = compute_orb_similarity(img1, img2)
        hist = compute_histogram_similarity(img1, img2)
        ssim = compute_ssim_similarity(img1, img2)
        score = float(np.clip(np.dot(self.weights, np.array([orb, hist, ssim], dtype=np.float32)), 0.0, 1.0))
        return score, orb, hist, ssim

    def match(self, query: Path, candidates: Iterable[Path]) -> List[MatchResult]:
        query_img = _load_image(query)
        results: List[MatchResult] = []
        for candidate in candidates:
            candidate_img = _load_image(candidate)
            score, orb, hist, ssim = self.score_pair(query_img, candidate_img)
            results.append(MatchResult(candidate, score, orb, hist, ssim))
        results.sort(key=lambda r: r.score, reverse=True)
        return results
