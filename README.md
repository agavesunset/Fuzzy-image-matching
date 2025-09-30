# Fuzzy Image Matching

A Python utility for fuzzy image matching using a combination of ORB feature matching, color histogram comparison, and Structural Similarity Index (SSIM).

## Features

- Compare a query image against one or more candidate images or directories.
- Combines multiple similarity metrics with configurable weights.
- Command line interface for quick usage.
- Minimal web UI for running comparisons without the terminal.

## Requirements

- Python 3.9+
- `opencv-python`
- `scikit-image`
- `numpy`

Install dependencies using pip:

```bash
pip install opencv-python scikit-image numpy
```

For the optional web interface install Flask as well:

```bash
pip install flask
```

## Usage

Run the matcher from the project root:

```bash
python -m fuzzy_image_matching.cli <query_image> <candidate1> [<candidate2> ...] [options]
```

You can pass directories as candidate arguments to compare against all files in the directory.

### Options

- `--weights ORB HIST SSIM`: Custom weights for ORB, histogram, and SSIM scores. Defaults to `0.4 0.3 0.3`.
- `--top N`: Limit the output to the top N matches (defaults to 5).

### Example

```bash
python -m fuzzy_image_matching.cli ./examples/query.jpg ./examples/candidates --top 3
```

This will output the top three matches and show the individual metric contributions.


## Web UI


Launch the browser-based interface with Flask:

```bash
python -m flask --app fuzzy_image_matching.webapp run
```

Then open <http://localhost:5000> to upload a query image and one or more candidate images. You can adjust the metric weights and number of results directly in the interface, and the page will display previews and detailed scores for the best matches.


## Project Structure

```
fuzzy_image_matching/
├── __init__.py
├── cli.py
└── matching.py
```

## License

MIT
