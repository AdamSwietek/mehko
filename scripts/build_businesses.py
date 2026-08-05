#!/usr/bin/env python3
"""Validate canonical business data and generate the public map JSON."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from business_data import (
    CANONICAL_PATH,
    PUBLIC_PATH,
    DataValidationError,
    load_businesses,
    serialized_public_data,
    write_businesses,
)


def import_geocodes(rows: list[dict[str, str]], geocode_path: Path) -> int:
    with geocode_path.open(newline="", encoding="utf-8") as handle:
        geocodes = {row["business_name"]: row for row in csv.DictReader(handle)}

    imported = 0
    for row in rows:
        geocode = geocodes.get(row["business_name"])
        if not geocode:
            continue
        before = (row["lat"], row["lon"], row["matched_address"])
        if not row["lat"]:
            row["lat"] = (geocode.get("lat") or "").strip()
        if not row["lon"]:
            row["lon"] = (geocode.get("lon") or "").strip()
        if not row["matched_address"]:
            row["matched_address"] = (geocode.get("matched_address") or "").strip()
        if before != (row["lat"], row["lon"], row["matched_address"]):
            imported += 1
    return imported


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if public JSON is stale")
    parser.add_argument("--validate-only", action="store_true", help="validate without writing JSON")
    parser.add_argument("--import-geocodes", type=Path, help="fill blank coordinates from a legacy CSV")
    args = parser.parse_args()

    rows = load_businesses()
    if args.import_geocodes:
        count = import_geocodes(rows, args.import_geocodes)
        write_businesses(rows)
        print(f"Imported geocodes for {count} businesses into {CANONICAL_PATH.relative_to(CANONICAL_PATH.parent.parent)}")

    try:
        rendered = serialized_public_data(rows)
    except DataValidationError as exc:
        print(f"Business data validation failed:\n{exc}", file=sys.stderr)
        return 1

    mapped = sum(bool(row["lat"] and row["lon"]) for row in rows)
    if args.check:
        current = PUBLIC_PATH.read_text(encoding="utf-8") if PUBLIC_PATH.exists() else ""
        if current != rendered:
            print("data/businesses.json is stale; run scripts/build_businesses.py", file=sys.stderr)
            return 1
        print(f"Validated {len(rows)} businesses; public JSON is current ({mapped} mapped).")
        return 0

    if args.validate_only:
        print(f"Validated {len(rows)} businesses ({mapped} mapped, {len(rows) - mapped} awaiting coordinates).")
        return 0

    PUBLIC_PATH.write_text(rendered, encoding="utf-8")
    print(f"Generated data/businesses.json ({mapped} mapped businesses from {len(rows)} canonical records).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
