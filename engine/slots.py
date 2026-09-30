"""``extract_slots(text)`` primitive — deterministic regex + corpus gazetteer.

Never generative: a slot value is always a literal span of what was said, so
slots can't invent information. Keys:

headcount, amount, percent, duration, date, location, entity, number
"""
from __future__ import annotations

import re

_NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    "twenty-five": 25, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100, "a hundred": 100,
}
_NUM = r"(\d[\d,]*(?:\.\d+)?|" + "|".join(sorted(map(re.escape, _NUM_WORDS), key=len, reverse=True)) + r")"
_MONTHS = ("january|february|march|april|may|june|july|august|september|october|november|december|"
           "jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec")

_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("amount", re.compile(r"(?:inr|rs\.?|₹|usd|\$|eur|€)\s?" + _NUM + r"(?:\s?(?:lakh|lakhs|k|thousand))?"
                          r"|" + _NUM + r"\s?(?:rupees|dollars|usd|inr|euros|lakh|lakhs)\b", re.I)),
    ("percent", re.compile(_NUM + r"\s?(?:%|percent\b)", re.I)),
    ("headcount", re.compile(_NUM + r"\s+(?:people|persons|person|attendees|guests|participants|pax|"
                             r"employees|members|heads|delegates|seats|covers)\b", re.I)),
    ("duration", re.compile(_NUM + r"\s+(?:days?|nights?|hours?|weeks?|months?)\b", re.I)),
    ("date", re.compile(r"\b(?:\d{1,2}(?:st|nd|rd|th)?\s+(?:" + _MONTHS + r")\b|(?:" + _MONTHS +
                        r")\s+\d{1,2}(?:st|nd|rd|th)?\b|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|"
                        r"(?:next|this|last)\s+(?:week|month|monday|tuesday|wednesday|thursday|friday|"
                        r"saturday|sunday)|tomorrow|today|yesterday)", re.I)),
]
SLOT_UNIT_WORDS = frozenset(
    "people persons person attendees guests participants pax employees members heads delegates seats covers "
    "days day nights night hours hour weeks week months month percent inr usd rs rupees dollars lakh lakhs".split())
_BARE_NUM = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")


def _norm_num(s: str) -> str:
    s = s.lower().strip()
    if s in _NUM_WORDS:
        return str(_NUM_WORDS[s])
    return s.replace(",", "")


def extract_slots(text: str, gazetteer: dict[str, tuple[str, str]] | None = None) -> dict[str, str]:
    slots: dict[str, list[str]] = {}
    consumed: list[tuple[int, int]] = []

    def add(key: str, val: str, span: tuple[int, int]):
        if any(a < span[1] and span[0] < b for a, b in consumed):
            return
        consumed.append(span)
        if val not in slots.setdefault(key, []):
            slots[key].append(val)

    for key, pat in _PATTERNS:
        for m in pat.finditer(text):
            if key in ("headcount", "percent", "duration"):
                num = _norm_num(m.group(1))
                unit = m.group(0)[len(m.group(1)):].strip().lower()
                val = num if key == "headcount" else f"{num} {unit}".strip()
            elif key == "amount":
                val = re.sub(r"\s+", " ", m.group(0).strip().lower())
            else:
                val = m.group(0).strip().lower()
            add(key, val, m.span())

    if gazetteer:
        low = text.lower()
        for phrase in sorted(gazetteer, key=len, reverse=True):
            for m in re.finditer(r"\b" + re.escape(phrase) + r"\b", low):
                surface, kind = gazetteer[phrase]
                add(kind, surface, m.span())

    for m in _BARE_NUM.finditer(text):
        add("number", m.group(0).replace(",", ""), m.span())

    return {k: ", ".join(v) for k, v in slots.items()}


def strip_slot_spans(text: str, gazetteer: dict[str, tuple[str, str]] | None = None) -> str:
    """The text with every span that became a slot removed (what is left is the actual ask)."""
    out = text
    for key, pat in _PATTERNS:
        if key != "date":
            out = pat.sub(" ", out)
    if gazetteer:
        low = out.lower()
        for phrase in sorted(gazetteer, key=len, reverse=True):
            for m in re.finditer(r"\b" + re.escape(phrase) + r"\b", low):
                out = out[:m.start()] + " " * (m.end() - m.start()) + out[m.end():]
            low = out.lower()
    return " ".join(out.split())


def slot_values(slots: dict[str, str]) -> list[str]:
    return [v for vals in slots.values() for v in vals.split(", ")]


def rewrite_with_slots(text: str, old: dict[str, str], new: dict[str, str]) -> str:
    """Substitute changed slot values inside previously spoken text ("30 people" -> "60 people")."""
    for key in changed_slots(old, new):
        first_new = new[key].split(", ")[0]
        for o in old[key].split(", "):
            text = re.sub(r"\b" + re.escape(o) + r"\b", first_new, text, flags=re.I)
    return text


def changed_slots(old: dict[str, str], new: dict[str, str]) -> set[str]:
    """Keys present in both whose values differ (the Tier-1 slot-diff signal)."""
    return {k for k in new if k in old and new[k] != old[k]}
