# MEHKO Map

The canonical business dataset is `data/mehko_businesses.csv`. It contains the
public listing fields, stable IDs, geocodes, and optional private contact email.
The website reads generated public data from `data/businesses.json`.

Do not manually synchronize classification, social, geocode, or public JSON
files. Update the canonical CSV through the command below; it validates the
whole dataset and regenerates the public JSON automatically.

## Add or update a submission

Save the submission as JSON using either snake-case keys or the labels from the
submission form:

```json
{
  "Submission type": "new",
  "Business name": "Example Coffee",
  "Address": "123 Example Ave",
  "City": "Los Angeles",
  "Business type": "Beverage/Coffee/Tea",
  "Instagram": "examplecoffee",
  "Website": "www.example.com",
  "Notes": "A home cafe serving coffee and pastries.",
  "Email": "owner@example.com"
}
```

Then run:

```bash
python3 scripts/upsert_business.py --submission submission.json
```

New addresses are geocoded through the U.S. Census geocoder. The command
normalizes addresses, Instagram handles, websites, and cuisine tags. Contact
email is retained in the canonical CSV but intentionally excluded from the
public JSON.

An existing record can also be corrected directly:

```bash
python3 scripts/upsert_business.py \
  --original-name GRANADA \
  --business-type "Beverage/Coffee/Tea" \
  --tags "middle eastern, mediterranean, hummus"
```

Use `--dry-run` to preview an update without writing files. Use `--no-geocode`
only when intentionally saving a record without map coordinates.

## Build and validate

Regenerate the public map data:

```bash
python3 scripts/build_businesses.py
```

Verify that the canonical data is valid and the generated JSON is current:

```bash
python3 scripts/build_businesses.py --check
```

The public builder excludes records without coordinates and never rewrites
`index.html`.

## Data ownership

- `data/mehko_businesses.csv`: authoritative source.
- `data/businesses.json`: generated public artifact.
- `data/business_classifications.json`, `data/business_social.json`, and
  `data/business_snippets.json`: legacy enrichment caches; they only fill blank
  canonical values in the notebook.
- `data/mehko_businesses_geocoded.csv`: legacy geocoding output; coordinates now
  live in the canonical CSV.
- `mehkho.ipynb`: exploratory and bulk-enrichment workflow, not the production
  update path.
