#!/usr/bin/env python3
import argparse
from pathlib import Path

DATASETS = {
    "UT_HAR": "Use existing project preparation or documented upstream source; raw data is not redistributed.",
    "NTU_Fi_HAR": (
        "Download NTU-Fi HAR from the official SenseFi processed-data source or Mendeley mirror. "
        "If the Mendeley ZIP has the known tail/CRC issue, recover with 7-Zip, place the recovered "
        "directory at data/raw/NTU-Fi_HAR, then run scripts/12_prepare_ntu_fi_recovered.py."
    ),
    "Widar3": "Download the SenseFi processed Widar ZIP; place/extract as data/raw/SenseFi_extracted/Widardata.",
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    Path(args.data_root).mkdir(parents=True, exist_ok=True)
    print(DATASETS[args.dataset])
    print(
        "Dry-run only: this script records placement and avoids redistributing restricted data."
        if args.dry_run
        else "Manual download/recovery required unless an official URL/token is configured."
    )
