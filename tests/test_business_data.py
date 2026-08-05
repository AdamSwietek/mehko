import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from business_data import (  # noqa: E402
    PUBLIC_PATH,
    load_businesses,
    normalize_instagram,
    normalize_website,
    public_payload,
    serialized_public_data,
    slugify,
    validate_businesses,
)
from upsert_business import load_submission  # noqa: E402


class BusinessDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_businesses()

    def test_canonical_data_is_valid(self):
        self.assertEqual(validate_businesses(self.rows), [])

    def test_ids_and_names_are_unique(self):
        ids = [row["business_id"] for row in self.rows]
        names = [row["business_name"].casefold() for row in self.rows]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(names), len(set(names)))

    def test_public_payload_excludes_private_contact_data(self):
        payload = public_payload(self.rows)
        self.assertTrue(payload["businesses"])
        self.assertTrue(all("contact_email" not in row for row in payload["businesses"]))
        cold_brew = next(row for row in payload["businesses"] if row["id"] == "cold-brew-coffee-crew")
        self.assertNotIn("proton.me", str(cold_brew))

    def test_public_json_is_current(self):
        self.assertEqual(PUBLIC_PATH.read_text(encoding="utf-8"), serialized_public_data(self.rows))

    def test_normalizers(self):
        self.assertEqual(slugify("Cold Brew Coffee Crew"), "cold-brew-coffee-crew")
        self.assertEqual(normalize_instagram("https://instagram.com/example/"), "@example")
        self.assertEqual(normalize_website("www.example.com"), "https://www.example.com")

    def test_submission_form_labels_are_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.json"
            path.write_text(
                '{"Submission type":"new","Business name":"Example Cafe",'
                '"Business type":"Beverage/Coffee/Tea","Notes":"Coffee",'
                '"Email":"owner@example.com"}',
                encoding="utf-8",
            )
            submission = load_submission(path)
        self.assertEqual(submission["submission_type"], "new")
        self.assertEqual(submission["business_name"], "Example Cafe")
        self.assertEqual(submission["description"], "Coffee")
        self.assertEqual(submission["contact_email"], "owner@example.com")


if __name__ == "__main__":
    unittest.main()
