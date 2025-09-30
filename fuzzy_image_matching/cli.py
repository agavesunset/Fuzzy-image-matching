"""Command line interface for fuzzy image matching."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


from .matching import FuzzyImageMatcher, ImageProcessingError



class ExistingPath(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        path = Path(values)
        if not path.exists():
            parser.error(f"Path does not exist: {path}")
        setattr(namespace, self.dest, path)


def _iter_candidate_paths(path: Path) -> Iterable[Path]:
    if path.is_dir():
        yield from sorted(p for p in path.iterdir() if p.is_file())
    else:
        yield path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fuzzy image matcher")
    parser.add_argument("query", action=ExistingPath, help="Query image")
    parser.add_argument(
        "candidates",
        nargs="+",
        help="Candidate image files or directories",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs=3,
        metavar=("ORB", "HIST", "SSIM"),
        help="Weights for similarity metrics (default: 0.4 0.3 0.3)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of top matches to show",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    matcher = FuzzyImageMatcher(weights=args.weights)
    candidate_files: list[Path] = []
    for candidate in args.candidates:
        candidate_files.extend(_iter_candidate_paths(Path(candidate)))


    try:
        results = matcher.match(args.query, candidate_files)
    except (FileNotFoundError, ImageProcessingError) as exc:
        parser.error(str(exc))


    top_n = args.top if args.top > 0 else len(results)

    print(f"Query image: {args.query}")
    print("Candidates scored (higher is better):")
    for result in results[:top_n]:
        print(
            f"  {result.candidate}: score={result.score:.3f} "
            f"(ORB={result.orb:.3f}, HIST={result.histogram:.3f}, SSIM={result.ssim:.3f})"
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
