"""Simple web UI for fuzzy image matching."""

from __future__ import annotations

import base64
from typing import List, Tuple

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from .matching import FuzzyImageMatcher, ImageProcessingError


DEFAULT_WEIGHTS: Tuple[float, float, float] = (0.4, 0.3, 0.3)


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        template_folder="templates",
    )
    # Limit uploads to 16 MiB to prevent extremely large files.
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

    @app.route("/", methods=["GET", "POST"])
    def index():
        weights = list(DEFAULT_WEIGHTS)
        errors: List[str] = []
        results: List[dict] | None = None
        query_preview = None
        query_image: np.ndarray | None = None
        top_value = 5

        if request.method == "POST":
            # Parse weights, keeping the provided values even if invalid to display back to the user.
            weight_fields = [
                ("weight_orb", DEFAULT_WEIGHTS[0], "ORB weight"),
                ("weight_hist", DEFAULT_WEIGHTS[1], "Histogram weight"),
                ("weight_ssim", DEFAULT_WEIGHTS[2], "SSIM weight"),
            ]
            parsed_weights: List[float] = []
            for idx, (field, default, label) in enumerate(weight_fields):
                value_raw = request.form.get(field, str(default)).strip()
                try:
                    value = float(value_raw)
                except (TypeError, ValueError):
                    errors.append(f"{label} must be a number.")
                    value = weights[idx]
                parsed_weights.append(value)
            weights = parsed_weights

            if sum(weights) <= 0:
                errors.append("At least one weight must be greater than zero.")

            top_raw = request.form.get("top", str(top_value)).strip()
            try:
                top_value = int(top_raw)
                if top_value <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                errors.append("Top results must be a positive integer.")
                top_value = 5

            query_file = request.files.get("query")
            if not query_file or not query_file.filename:
                errors.append("Please upload a query image.")
            else:
                try:
                    query_image = _file_to_image(query_file)
                    query_preview = _image_to_data_url(query_image)
                except ValueError:
                    errors.append("Could not read the query image. Please upload a valid image file.")

            candidate_files = [
                f for f in request.files.getlist("candidates") if f and f.filename
            ]
            if not candidate_files:
                errors.append("Please upload at least one candidate image.")

            if not errors and query_image is not None:
                try:
                    matcher = FuzzyImageMatcher(weights=weights)
                except ValueError as exc:
                    errors.append(str(exc))
                    matcher = None

                if matcher is not None:
                    scored_results = []
                    for storage in candidate_files:
                        try:
                            candidate_image = _file_to_image(storage)
                        except ValueError:
                            errors.append(
                                f"Could not read candidate image: {storage.filename or 'Unnamed file'}."
                            )
                            continue

                        try:
                            score, orb, hist, ssim = matcher.score_pair(query_image, candidate_image)
                        except ImageProcessingError as exc:
                            errors.append(
                                f"Failed to compare {storage.filename or 'candidate image'}: {exc}"
                            )
                            continue
                        except Exception as exc:  # pragma: no cover - defensive
                            app.logger.exception("Unexpected error while scoring image")
                            errors.append(
                                f"An unexpected error occurred while scoring {storage.filename or 'a candidate image'}."
                            )
                            continue

                        try:
                            preview = _image_to_data_url(candidate_image)
                        except ValueError:
                            errors.append(
                                f"Could not generate preview for {storage.filename or 'a candidate image'}."
                            )
                            preview = None

                        scored_results.append(
                            {
                                "name": storage.filename or "Candidate",
                                "score": float(score),
                                "orb": float(orb),
                                "histogram": float(hist),
                                "ssim": float(ssim),
                                "preview": preview,
                            }
                        )

                    if scored_results:
                        scored_results.sort(key=lambda item: item["score"], reverse=True)
                        limit = top_value if top_value > 0 else len(scored_results)
                        results = scored_results[:limit]
                    else:
                        results = []

        payload = {
            "errors": errors,
            "weights": [float(value) for value in weights],
            "top": top_value,
            "query_preview": query_preview,
            "results": results or [],
            "submitted": request.method == "POST",
        }
        wants_json = request.method == "POST" and request.accept_mimetypes.best == "application/json"
        if wants_json:
            status_code = 200 if not errors else 400
            return jsonify(payload), status_code

        return render_template(
            "index.html",
            weights=weights,
            top=top_value,
            results=results if request.method == "POST" else None,
            errors=errors,
            query_preview=query_preview,
            submitted=request.method == "POST",
            placeholder_text="Results will appear here after you compare images.",
            initial_payload=payload,
        )

    return app


def _file_to_image(storage) -> np.ndarray:
    """Decode an uploaded file into a BGR OpenCV image."""
    storage.stream.seek(0)
    file_bytes = storage.read()
    if not file_bytes:
        raise ValueError("Empty upload")
    array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image data")
    return image


def _image_to_data_url(image: np.ndarray) -> str:
    """Convert an image array to a base64 data URL for inline display."""
    success, buffer = cv2.imencode(".jpg", image)
    if not success:
        raise ValueError("Unable to encode image")
    encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    app.run(debug=True)
