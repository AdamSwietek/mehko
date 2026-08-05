#!/usr/bin/env python3
"""Add or update one business in the canonical dataset, then rebuild map data."""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

from business_data import (
    CANONICAL_FIELDS,
    CANONICAL_PATH,
    DataValidationError,
    load_businesses,
    normalize_instagram,
    normalize_website,
    require_valid,
    slugify,
    write_businesses,
    write_public_data,
)


ALIASES = {
    "submission_type": "submission_type",
    "business_id": "business_id",
    "original_name": "original_name",
    "business_name": "business_name",
    "name": "business_name",
    "address": "address",
    "city": "city",
    "postal_code": "postal_code",
    "zip": "postal_code",
    "zip_code": "postal_code",
    "business_type": "business_type",
    "cuisine_tags": "cuisine_tags",
    "tags": "cuisine_tags",
    "description": "description",
    "notes": "description",
    "instagram": "instagram",
    "website": "website",
    "email": "contact_email",
    "contact_email": "contact_email",
    "social_confidence": "social_confidence",
    "lat": "lat",
    "latitude": "lat",
    "lon": "lon",
    "longitude": "lon",
    "matched_address": "matched_address",
}


def normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def load_submission(path: Path) -> dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("submission JSON must be an object")
    result = {}
    for key, value in raw.items():
        alias = ALIASES.get(normalized_key(key))
        if alias and value is not None:
            result[alias] = str(value).strip()
    return result


def geocode(data: dict[str, str]) -> dict[str, str]:
    query = ", ".join(
        part for part in (data.get("address"), data.get("city"), "CA", data.get("postal_code")) if part
    )
    params = urllib.parse.urlencode(
        {"address": query, "benchmark": "Public_AR_Current", "format": "json"}
    )
    url = f"https://geocoding.geo.census.gov/geocoder/locations/onelineaddress?{params}"
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.load(response)
    matches = payload.get("result", {}).get("addressMatches", [])
    if not matches:
        raise ValueError(f"Census geocoder found no match for {query}")
    match = matches[0]
    components = match.get("addressComponents", {})
    return {
        "postal_code": str(components.get("zip") or data.get("postal_code") or ""),
        "lat": str(match["coordinates"]["y"]),
        "lon": str(match["coordinates"]["x"]),
        "matched_address": match.get("matchedAddress", ""),
    }


def find_existing(rows: list[dict[str, str]], data: dict[str, str]) -> dict[str, str] | None:
    business_id = data.get("business_id", "").casefold()
    lookup_name = (data.get("original_name") or data.get("business_name") or "").casefold()
    for row in rows:
        if business_id and row["business_id"].casefold() == business_id:
            return row
        if lookup_name and row["business_name"].casefold() == lookup_name:
            return row
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, help="JSON submission file")
    parser.add_argument("--submission-type", choices=("new", "update"))
    parser.add_argument("--id", dest="business_id")
    parser.add_argument("--original-name")
    parser.add_argument("--name", dest="business_name")
    parser.add_argument("--address")
    parser.add_argument("--city")
    parser.add_argument("--postal-code")
    parser.add_argument("--business-type")
    parser.add_argument("--tags", dest="cuisine_tags")
    parser.add_argument("--description")
    parser.add_argument("--instagram")
    parser.add_argument("--website")
    parser.add_argument("--email", dest="contact_email")
    parser.add_argument("--social-confidence", choices=("low", "medium", "high"))
    parser.add_argument("--lat")
    parser.add_argument("--lon")
    parser.add_argument("--matched-address")
    parser.add_argument("--no-geocode", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data = load_submission(args.submission) if args.submission else {}
    for key, value in vars(args).items():
        if key not in {"submission", "no_geocode", "dry_run"} and value is not None:
            data[key] = value

    rows = load_businesses()
    existing = find_existing(rows, data)
    submission_type = data.get("submission_type") or ("update" if existing else "new")
    if submission_type == "new" and existing:
        print(f"Business already exists: {existing['business_name']} ({existing['business_id']})", file=sys.stderr)
        return 1
    if submission_type == "update" and not existing:
        print("Could not find the business to update; provide --id or --original-name", file=sys.stderr)
        return 1

    is_new = existing is None
    row = {field: "" for field in CANONICAL_FIELDS} if is_new else existing.copy()
    old_address = (row["address"], row["city"], row["postal_code"])
    editable = set(CANONICAL_FIELDS) - {"updated_at"}
    for key, value in data.items():
        if key in editable:
            row[key] = value.strip()

    if not row["business_id"]:
        row["business_id"] = slugify(row["business_name"])
    row["address"] = row["address"].upper()
    row["city"] = row["city"].upper()
    row["instagram"] = normalize_instagram(row["instagram"])
    row["website"] = normalize_website(row["website"])
    row["cuisine_tags"] = ", ".join(tag.strip().lower() for tag in row["cuisine_tags"].split(",") if tag.strip())
    if (data.get("instagram") or data.get("website")) and not data.get("social_confidence"):
        row["social_confidence"] = "high"
    row["updated_at"] = date.today().isoformat()

    address_changed = old_address != (row["address"], row["city"], row["postal_code"])
    needs_geocode = not row["lat"] or not row["lon"] or address_changed
    if needs_geocode and not args.no_geocode and not (data.get("lat") and data.get("lon")):
        try:
            row.update(geocode(row))
        except Exception as exc:
            print(f"Geocoding failed: {exc}", file=sys.stderr)
            return 1

    if is_new:
        rows.append(row)
    else:
        rows[rows.index(existing)] = row

    try:
        require_valid(rows)
    except DataValidationError as exc:
        print(f"Business data validation failed:\n{exc}", file=sys.stderr)
        return 1

    action = "Add" if is_new else "Update"
    print(f"{action}: {row['business_name']} ({row['business_id']})")
    if args.dry_run:
        print(json.dumps(row, indent=2))
        return 0

    write_businesses(rows)
    write_public_data(rows)
    print(f"Updated {CANONICAL_PATH.relative_to(CANONICAL_PATH.parent.parent)} and data/businesses.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
