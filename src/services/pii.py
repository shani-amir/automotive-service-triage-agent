import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PiiMaskResult:
    masked_text: str
    redacted_types: list[str]


# Match formatted cards as groups to avoid crossing unrelated digit sequences.
_EMAIL_LOCAL = r"[\w.!#$%&'*+/=?^_`{|}~-]"
_PATTERNS = (
    ("email", rf"(?<!{_EMAIL_LOCAL}){_EMAIL_LOCAL}+@[\w-]+(?:\.[\w-]+)+\b"),
    ("credit_card", r"(?<!\d)(?:\d{4}([ -])\d{4}\1\d{4}\1\d{4}|\d{4}([ -])\d{6}\2\d{5}|\d{13,19})(?!\d)"),
    ("phone", r"(?<!\d)(?:05\d|\+972[ -]?5\d)[ -]?\d{3}[ -]?\d{4}(?!\d)"),
    ("israeli_id", r"(?<!\d)\d{9}(?!\d)"),
)


def mask_pii(text: str) -> PiiMaskResult:
    redacted_types = []
    for pii_type, pattern in _PATTERNS:
        text, count = re.subn(pattern, f"[{pii_type.upper()}]", text)
        if count:
            redacted_types.append(pii_type)
    return PiiMaskResult(masked_text=text, redacted_types=redacted_types)
