#!/usr/bin/env python3
"""
Download the tabular strawberry Zenodo files used by the case study.

This script is intentionally robust for repeatable portfolio runs:
- If the required local files already exist, it does not need Zenodo to be online.
- If Zenodo returns a transient 5xx/timeout error but local files exist, it continues with a warning.
- If files are missing, it retries the Zenodo API/downloads before failing clearly.

Run from the repository root:
    python scripts/00_download_strawberry_zenodo.py
"""

from __future__ import annotations

import time
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import pandas as pd
import requests
from requests import RequestException


RECORD_ID = "10957909"
API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"

OUTDIR = Path("data/raw/strawberry_zenodo")
OUTDIR.mkdir(parents=True, exist_ok=True)

TARGET_FILES = {
    "measurements.zip",
    "datasetProcessing.py",
}

MAX_RETRIES = 3
REQUEST_TIMEOUT_SECONDS = 60


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def human_size(n_bytes: int | float) -> str:
    size = float(n_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"



def local_target_files() -> dict[str, Path]:
    return {filename: OUTDIR / filename for filename in TARGET_FILES}



def have_local_required_files() -> bool:
    files = local_target_files()
    return all(path.exists() and path.stat().st_size > 0 for path in files.values())



def existing_measurement_csvs() -> list[Path]:
    measurement_root = OUTDIR / "measurements"
    if not measurement_root.exists():
        return []
    return sorted(measurement_root.rglob("*.csv"))



def get_with_retries(url: str, *, stream: bool = False) -> requests.Response:
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                url,
                stream=stream,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            return response
        except RequestException as exc:
            last_error = exc
            wait_seconds = attempt * 5
            print(
                f"Request failed on attempt {attempt}/{MAX_RETRIES}: {exc}. "
                f"Retrying in {wait_seconds}s..." if attempt < MAX_RETRIES else f"Request failed on final attempt: {exc}"
            )
            if attempt < MAX_RETRIES:
                time.sleep(wait_seconds)

    raise RuntimeError(f"Could not retrieve {url}") from last_error


# ---------------------------------------------------------------------------
# Zenodo metadata / downloads
# ---------------------------------------------------------------------------


def fetch_zenodo_record() -> dict | None:
    """Fetch Zenodo metadata, returning None if local files are enough and Zenodo is unavailable."""
    try:
        response = get_with_retries(API_URL)
        return response.json()
    except Exception as exc:
        if have_local_required_files():
            print(
                "WARNING: Could not reach Zenodo API, but required local files already exist. "
                "Continuing with local files."
            )
            print(f"Zenodo error: {exc}")
            return None
        raise RuntimeError(
            "Could not reach Zenodo and required local files are missing. "
            "Try again later or manually place measurements.zip and datasetProcessing.py under "
            f"{OUTDIR}."
        ) from exc



def write_manifest(files: list[dict]) -> pd.DataFrame:
    manifest_rows = []

    print("Zenodo files:")
    for file_info in files:
        filename = file_info["key"]
        size = file_info.get("size", 0)
        download_url = file_info["links"]["self"]

        manifest_rows.append(
            {
                "filename": filename,
                "size_bytes": size,
                "size_human": human_size(size),
                "download_url": download_url,
            }
        )
        print(f"- {filename}: {human_size(size)}")

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(OUTDIR / "zenodo_file_manifest.csv", index=False)
    return manifest



def download_file(url: str, outpath: Path) -> None:
    if outpath.exists() and outpath.stat().st_size > 0:
        print(f"Already exists: {outpath}")
        return

    tmp_path = outpath.with_suffix(outpath.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    print(f"Downloading: {outpath.name}")
    response = get_with_retries(url, stream=True)

    with open(tmp_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    tmp_path.replace(outpath)



def download_target_files_from_record(record: dict | None) -> None:
    if record is None:
        for filename, path in local_target_files().items():
            print(f"Using local file: {path}" if path.exists() else f"Missing local file: {path}")
        return

    files = record["files"]
    write_manifest(files)

    for file_info in files:
        filename = file_info["key"]

        if filename not in TARGET_FILES:
            print(f"Skipping large/secondary file for now: {filename}")
            continue

        download_file(file_info["links"]["self"], OUTDIR / filename)


# ---------------------------------------------------------------------------
# Extraction / inspection
# ---------------------------------------------------------------------------


def extract_measurements_if_needed() -> Path:
    measurements_zip = OUTDIR / "measurements.zip"
    measurement_dir = OUTDIR / "measurements"

    if existing_measurement_csvs():
        print(f"Measurements already extracted under: {measurement_dir}")
        return measurement_dir

    if not measurements_zip.exists():
        raise FileNotFoundError(
            f"Could not find {measurements_zip}. Run this script again when Zenodo is available."
        )

    measurement_dir.mkdir(parents=True, exist_ok=True)
    try:
        with ZipFile(measurements_zip, "r") as zip_ref:
            zip_ref.extractall(measurement_dir)
    except BadZipFile as exc:
        raise RuntimeError(f"Downloaded file is not a valid zip: {measurements_zip}") from exc

    print(f"Extracted measurements to: {measurement_dir}")
    return measurement_dir



def inspect_measurements(measurement_dir: Path) -> None:
    csv_files = sorted(measurement_dir.rglob("*.csv"))

    if not csv_files:
        print("No CSV files found after extraction.")
        return

    print("\nCSV files found:")
    for path in csv_files:
        df = pd.read_csv(path)
        print("\n" + "=" * 80)
        print(path.relative_to(measurement_dir))
        print(f"shape: {df.shape}")
        print("columns:")
        print(list(df.columns))
        print("\nfirst rows:")
        print(df.head(3))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    if have_local_required_files():
        print("Required raw files already exist locally. Zenodo API is optional for this run.")

    record = fetch_zenodo_record()
    download_target_files_from_record(record)

    measurement_dir = extract_measurements_if_needed()
    inspect_measurements(measurement_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
