from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import requests


RECORD_ID = "10957909"
API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"

OUTDIR = Path("data/raw/strawberry_zenodo")
OUTDIR.mkdir(parents=True, exist_ok=True)

TARGET_FILES = {
    "measurements.zip",
    "datasetProcessing.py",
}


def human_size(n_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def download_file(url: str, outpath: Path):
    if outpath.exists():
        print(f"Already exists: {outpath}")
        return

    print(f"Downloading: {outpath.name}")

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()

        with open(outpath, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def inspect_measurements(measurement_dir: Path):
    csv_files = sorted(measurement_dir.rglob("*.csv"))

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


def main():
    response = requests.get(API_URL, timeout=60)
    response.raise_for_status()
    record = response.json()

    files = record["files"]

    manifest_rows = []

    print("Zenodo files:")
    for file_info in files:
        filename = file_info["key"]
        size = file_info["size"]

        manifest_rows.append(
            {
                "filename": filename,
                "size_bytes": size,
                "size_human": human_size(size),
                "download_url": file_info["links"]["self"],
            }
        )

        print(f"- {filename}: {human_size(size)}")

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(OUTDIR / "zenodo_file_manifest.csv", index=False)

    for file_info in files:
        filename = file_info["key"]

        if filename not in TARGET_FILES:
            print(f"Skipping large/secondary file for now: {filename}")
            continue

        outpath = OUTDIR / filename
        download_file(file_info["links"]["self"], outpath)

    measurements_zip = OUTDIR / "measurements.zip"
    measurement_dir = OUTDIR / "measurements"

    if measurements_zip.exists():
        measurement_dir.mkdir(exist_ok=True)

        with ZipFile(measurements_zip, "r") as zip_ref:
            zip_ref.extractall(measurement_dir)

        print(f"\nExtracted measurements to: {measurement_dir}")
        inspect_measurements(measurement_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
