from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "reference"
OUTPUT_PATH = OUTPUT_DIR / "taxi_zone_lookup.csv"

SOURCE_URL = (
    "https://d37ci6vzurychx.cloudfront.net/misc/"
    "taxi_zone_lookup.csv"
)


def download_zone_lookup() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading NYC Taxi Zone Lookup...")

    response = requests.get(
        SOURCE_URL,
        timeout=60,
    )
    response.raise_for_status()

    OUTPUT_PATH.write_bytes(response.content)

    df = pd.read_csv(OUTPUT_PATH)

    required_columns = [
        "LocationID",
        "Borough",
        "Zone",
        "service_zone",
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        OUTPUT_PATH.unlink()
        raise ValueError(f"Missing columns: {missing}")

    print("\nDownload completed")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Saved to: {OUTPUT_PATH}")

    print("\nFirst 10 rows:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    download_zone_lookup()