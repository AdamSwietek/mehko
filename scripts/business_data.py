"""Canonical MEHKO business data loading, validation, and publishing."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PATH = PROJECT_ROOT / "data" / "mehko_businesses.csv"
PUBLIC_PATH = PROJECT_ROOT / "data" / "businesses.json"

CANONICAL_FIELDS = [
    "business_id",
    "business_name",
    "address",
    "city",
    "postal_code",
    "business_type",
    "cuisine_tags",
    "description",
    "instagram",
    "website",
    "contact_email",
    "social_confidence",
    "lat",
    "lon",
    "matched_address",
    "updated_at",
]

BUSINESS_TYPES = {
    "Bakery",
    "BBQ",
    "Beverage/Coffee/Tea",
    "Breakfast/Brunch",
    "Caribbean",
    "Catering/Events",
    "Charcuterie/Boards",
    "Chinese",
    "Comfort Food/Soul Food",
    "Desserts/Sweets",
    "Filipino",
    "Indian/South Asian",
    "Japanese/Korean",
    "Latin American",
    "Mediterranean/Middle Eastern",
    "Mexican",
    "Nigerian/West African",
    "Other",
    "Pizza/Italian",
    "Salvadoran/Guatemalan",
    "Sandwiches/Burgers",
    "Seafood",
    "Vegan/Vegetarian",
    "Vietnamese",
}


class DataValidationError(ValueError):
    """Raised when canonical business data is invalid."""


def slugify(value: str) -> str:
    ascii_value = (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    return re.sub(r"[^a-z0-9]+", "-", ascii_value).strip("-")


def normalize_instagram(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    value = re.sub(r"^https?://(?:www\.)?instagram\.com/", "", value, flags=re.I)
    value = value.strip("/@ ")
    return f"@{value}" if value else ""


def normalize_website(value: str) -> str:
    value = value.strip()
    if value and not re.match(r"^https?://", value, flags=re.I):
        value = f"https://{value}"
    return value


def load_businesses(path: Path = CANONICAL_PATH) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    normalized = []
    for source in rows:
        row = {field: (source.get(field) or "").strip() for field in CANONICAL_FIELDS}
        if not row["business_id"]:
            row["business_id"] = slugify(row["business_name"])
        normalized.append(row)
    return normalized


def write_businesses(rows: list[dict[str, str]], path: Path = CANONICAL_PATH) -> None:
    rows = sorted(rows, key=lambda row: row["business_name"].casefold())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in CANONICAL_FIELDS} for row in rows)


def validate_businesses(rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    seen_ids: dict[str, int] = {}
    seen_names: dict[str, int] = {}
    required = ("business_id", "business_name", "address", "city", "postal_code", "business_type")

    for line_number, row in enumerate(rows, start=2):
        label = row.get("business_name") or f"row {line_number}"
        for field in required:
            if not row.get(field, "").strip():
                errors.append(f"{label}: missing {field}")

        business_id = row.get("business_id", "")
        name_key = row.get("business_name", "").casefold()
        if business_id in seen_ids:
            errors.append(f"{label}: duplicate business_id also used on row {seen_ids[business_id]}")
        else:
            seen_ids[business_id] = line_number
        if name_key in seen_names:
            errors.append(f"{label}: duplicate business_name also used on row {seen_names[name_key]}")
        else:
            seen_names[name_key] = line_number

        if row.get("business_type") not in BUSINESS_TYPES:
            errors.append(f"{label}: unsupported business_type {row.get('business_type')!r}")
        if business_id and not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", business_id):
            errors.append(f"{label}: business_id must be a lowercase slug")
        if row.get("postal_code") and not re.fullmatch(r"\d{5}", row["postal_code"]):
            errors.append(f"{label}: postal_code must contain five digits")
        if row.get("social_confidence") not in {"", "low", "medium", "high"}:
            errors.append(f"{label}: invalid social_confidence")
        if row.get("instagram") and not row["instagram"].startswith("@"):
            errors.append(f"{label}: instagram must start with @")
        if row.get("website") and not re.match(r"^https?://", row["website"], flags=re.I):
            errors.append(f"{label}: website must start with http:// or https://")
        if row.get("contact_email") and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", row["contact_email"]):
            errors.append(f"{label}: contact_email is invalid")

        lat_text, lon_text = row.get("lat", ""), row.get("lon", "")
        if bool(lat_text) != bool(lon_text):
            errors.append(f"{label}: lat and lon must either both be set or both be blank")
        elif lat_text and lon_text:
            try:
                lat, lon = float(lat_text), float(lon_text)
                if not (32.0 <= lat <= 35.5 and -120.0 <= lon <= -116.0):
                    errors.append(f"{label}: coordinates are outside Southern California")
            except ValueError:
                errors.append(f"{label}: lat and lon must be numbers")

    return errors


def require_valid(rows: list[dict[str, str]]) -> None:
    errors = validate_businesses(rows)
    if errors:
        raise DataValidationError("\n".join(f"- {error}" for error in errors))


def public_payload(rows: list[dict[str, str]]) -> dict:
    businesses = []
    for row in rows:
        if not row["lat"] or not row["lon"]:
            continue
        businesses.append(
            {
                "id": row["business_id"],
                "name": row["business_name"],
                "type": row["business_type"],
                "tags": row["cuisine_tags"],
                "description": row["description"],
                "address": f"{row['address']}, {row['city']} {row['postal_code']}",
                "instagram": row["instagram"],
                "website": row["website"],
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
            }
        )
    return {
        "businesses": businesses,
        "types": sorted({row["business_type"] for row in rows if row["business_type"]}),
    }


def serialized_public_data(rows: list[dict[str, str]]) -> str:
    require_valid(rows)
    return json.dumps(public_payload(rows), indent=2, ensure_ascii=False) + "\n"


def write_public_data(rows: list[dict[str, str]], path: Path = PUBLIC_PATH) -> None:
    path.write_text(serialized_public_data(rows), encoding="utf-8")
