from unittest import TestCase

from src.services.pii import mask_pii


class TestMaskPii(TestCase):
    def test_masks_pii_without_changing_plates(self):
        cases = [
            ("123456782", "israeli_id"),
            ("052-555-0199", "phone"),
            ("0501234567", "phone"),
            ("+972-52-555-0199", "phone"),
            ("customer+triage@example.test", "email"),
            ("4111 1111 1111 1111", "credit_card"),
            ("4111111111111111", "credit_card"),
            ("3782-822463-10005", "credit_card"),
        ]
        plates = "482-91-302, 12-345-67, 48291302, 1234567"
        for value, pii_type in cases:
            with self.subTest(value=value):
                result = mask_pii(f"{value} | {plates}")
                self.assertEqual(result.masked_text, f"[{pii_type.upper()}] | {plates}")
                self.assertEqual(result.redacted_types, [pii_type])


    def test_reports_each_redaction_type_once(self):
        result = mask_pii("123456782 052-5550199 customer@example.test 4111-1111-1111-1111 123456782")
        self.assertEqual(result.masked_text, "[ISRAELI_ID] [PHONE] [EMAIL] [CREDIT_CARD] [ISRAELI_ID]")
        self.assertCountEqual(result.redacted_types, ["israeli_id", "phone", "email", "credit_card"])
