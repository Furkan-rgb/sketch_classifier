#!/usr/bin/env python3
"""Recreate the complete local Quick, Draw! data directory.

By default this downloads all 345 categories in both formats used by this
workspace: 28x28 NumPy bitmaps and simplified stroke NDJSON files. The script
uses only the Python standard library and safely skips valid existing files.

Examples:

    python quickdraw_data/fetch.py
    python quickdraw_data/fetch.py --format bitmaps
    python quickdraw_data/fetch.py --format strokes --workers 2
    python quickdraw_data/fetch.py --category flower --category sun
    python quickdraw_data/fetch.py --force
"""

import argparse
import concurrent.futures
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "quickdraw_data"
CATEGORIES_URL = (
    "https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/"
    "master/categories.txt"
)
CHUNK_SIZE = 1024 * 1024
NUMPY_MAGIC = b"\x93NUMPY"

FORMATS = {
    "bitmaps": {
        "directory": DATA_DIRECTORY / "numpy_bitmap",
        "base_url": (
            "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"
        ),
        "suffix": ".npy",
    },
    "strokes": {
        "directory": DATA_DIRECTORY / "strokes",
        "base_url": (
            "https://storage.googleapis.com/quickdraw_dataset/full/simplified"
        ),
        "suffix": ".ndjson",
    },
}


def fetch_categories():
    request = urllib.request.Request(
        CATEGORIES_URL,
        headers={"User-Agent": "sketch-classifier-dataset-fetcher/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        categories = [
            line.strip()
            for line in response.read().decode("utf-8").splitlines()
            if line.strip()
        ]

    if len(categories) != 345 or len(set(categories)) != len(categories):
        raise ValueError(
            "The official category list did not contain 345 unique categories."
        )
    return categories


def is_valid_file(path, data_format):
    try:
        if path.stat().st_size == 0:
            return False
        with path.open("rb") as file:
            prefix = file.read(64)
        if data_format == "bitmaps":
            return prefix.startswith(NUMPY_MAGIC)
        return prefix.lstrip().startswith(b"{")
    except OSError:
        return False


def download_file(category, data_format, force=False):
    config = FORMATS[data_format]
    suffix = config["suffix"]
    destination = config["directory"] / f"{category}{suffix}"
    if destination.exists() and not force and is_valid_file(destination, data_format):
        return data_format, category, "skipped", destination.stat().st_size

    encoded_category = urllib.parse.quote(category, safe="")
    url = f"{config['base_url']}/{encoded_category}{suffix}"
    temporary = destination.with_name(f"{destination.name}.part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "sketch-classifier-dataset-fetcher/1.0"},
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with temporary.open("wb") as output:
                while chunk := response.read(CHUNK_SIZE):
                    output.write(chunk)

        if not is_valid_file(temporary, data_format):
            raise ValueError(f"downloaded {data_format} file failed validation")

        os.replace(temporary, destination)
        return data_format, category, "downloaded", destination.stat().st_size
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def format_size(byte_count):
    size = float(byte_count)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download all 345 Quick, Draw! categories as bitmaps and/or strokes."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("all", "bitmaps", "strokes"),
        default="all",
        help="Dataset format to fetch (default: all).",
    )
    parser.add_argument(
        "--category",
        action="append",
        dest="categories",
        help="Fetch one category; repeat to select several. Defaults to all 345.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel downloads (default: 4).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download again even when a valid local file exists.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all official categories without downloading.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    all_categories = fetch_categories()

    if args.list:
        print("\n".join(all_categories))
        return

    if not 1 <= args.workers <= 16:
        raise ValueError("--workers must be between 1 and 16.")

    requested = args.categories or all_categories
    unknown = [category for category in requested if category not in all_categories]
    if unknown:
        raise ValueError("Unknown categories: " + ", ".join(unknown))

    requested_set = set(requested)
    categories = [
        category for category in all_categories if category in requested_set
    ]
    data_formats = list(FORMATS) if args.format == "all" else [args.format]
    for data_format in data_formats:
        FORMATS[data_format]["directory"].mkdir(parents=True, exist_ok=True)

    tasks = [
        (category, data_format)
        for data_format in data_formats
        for category in categories
    ]
    print(
        f"Preparing {len(tasks)} files for {len(categories)} categories "
        f"({', '.join(data_formats)}) with {args.workers} workers."
    )
    if len(categories) == 345 and len(data_formats) == 2:
        print("The complete download requires approximately 59 GB of disk space.")

    failures = []
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_file, category, data_format, args.force): (
                data_format,
                category,
            )
            for category, data_format in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            data_format, category = futures[future]
            completed += 1
            try:
                _, _, status, byte_count = future.result()
                print(
                    f"[{completed:03d}/{len(tasks):03d}] "
                    f"{status:10} {data_format:7} {category} "
                    f"({format_size(byte_count)})"
                )
            except Exception as error:
                failures.append((data_format, category, error))
                print(
                    f"[{completed:03d}/{len(tasks):03d}] failed     "
                    f"{data_format:7} {category}: {error}",
                    file=sys.stderr,
                )

    if failures:
        failed_names = ", ".join(
            f"{data_format}/{category}"
            for data_format, category, _ in failures
        )
        raise RuntimeError(
            f"Failed to download {len(failures)} files: {failed_names}"
        )

    print("Complete Quick, Draw! dataset is ready in quickdraw_data/.")


if __name__ == "__main__":
    main()
