"""Corpus loading and chunking with ``Doc_ID §Section`` provenance.

Supported inputs (so a held-out corpus can be dropped in unchanged):

* ``*.md`` / ``*.txt`` — optional first-line header ``# Doc_12: Title``; sections
  start with ``## §N Heading``. Without ``§`` markers, ``##`` headings are
  numbered §1, §2, ...; without any headings, blank-line paragraphs become sections.
  The doc id falls back to the file stem.
* ``*.jsonl`` / ``*.json`` — records ``{"doc_id", "section", "text", "title"?}``.

Each section is one chunk. Sections longer than ``max_words`` are split on
sentence boundaries into ``§N.1``, ``§N.2`` ... so every chunk id is still a
real, verifiable locator.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from engine.primitives import split_sentences

_HEADER_RE = re.compile(r"^#\s+([A-Za-z]+[_-]?\d+)\s*[:\-—]\s*(.+)$")
_SECTION_RE = re.compile(r"^##\s+(§\s*[\w.]+)?\s*(.*)$")


@dataclass
class Chunk:
    idx: int
    doc_id: str
    section: str
    doc_title: str
    heading: str
    text: str
    sentences: list[str] = field(default_factory=list)

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id} {self.section}"

    @property
    def embed_text(self) -> str:
        return f"{self.doc_title}. {self.heading}. {self.text}"


def _split_long(section: str, text: str, max_words: int) -> list[tuple[str, str]]:
    if len(text.split()) <= max_words:
        return [(section, text)]
    parts, cur = [], []
    for s in split_sentences(text):
        if cur and len(" ".join(cur + [s]).split()) > max_words:
            parts.append(" ".join(cur))
            cur = []
        cur.append(s)
    if cur:
        parts.append(" ".join(cur))
    return [(f"{section}.{i + 1}", p) for i, p in enumerate(parts)]


def _parse_markdown(path: Path) -> list[tuple[str, str, str, str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    doc_id, doc_title = path.stem, path.stem.replace("_", " ")
    if lines and (m := _HEADER_RE.match(lines[0].strip())):
        doc_id, doc_title = m.group(1), m.group(2).strip()
        lines = lines[1:]
    elif lines and lines[0].startswith("# "):
        doc_title = lines[0][2:].strip()
        lines = lines[1:]

    sections: list[list] = []  # [section, heading, [lines]]
    auto = 0
    for line in lines:
        m = _SECTION_RE.match(line.strip())
        if m:
            auto += 1
            sec = (m.group(1) or f"§{auto}").replace(" ", "")
            sections.append([sec, m.group(2).strip(), []])
        elif sections:
            sections[-1][2].append(line)
        elif line.strip():
            sections.append([None, "", [line]])
    out = []
    if all(s[0] is None for s in sections):  # no headings at all -> paragraphs
        body = "\n".join(l for s in sections for l in s[2])
        for i, para in enumerate(p for p in re.split(r"\n\s*\n", body) if p.strip()):
            out.append((doc_id, f"§{i + 1}", doc_title, "", " ".join(para.split())))
        return out
    for sec, heading, body in sections:
        text = " ".join(" ".join(body).split())
        if text:
            out.append((doc_id, sec or "§0", doc_title, heading, text))
    return out


def _parse_json(path: Path) -> list[tuple[str, str, str, str, str]]:
    raw = path.read_text(encoding="utf-8")
    recs = [json.loads(l) for l in raw.splitlines() if l.strip()] if path.suffix == ".jsonl" else json.loads(raw)
    out = []
    for r in recs:
        sec = str(r.get("section", "§1"))
        sec = sec if sec.startswith("§") else f"§{sec}"
        out.append((str(r["doc_id"]), sec, r.get("title", str(r["doc_id"])), r.get("heading", ""), r["text"]))
    return out


def load_corpus(corpus_dir: str | Path, max_words: int = 180) -> list[Chunk]:
    corpus_dir = Path(corpus_dir)
    records = []
    for path in sorted(corpus_dir.rglob("*")):
        if path.suffix in (".md", ".txt"):
            records.extend(_parse_markdown(path))
        elif path.suffix in (".json", ".jsonl"):
            records.extend(_parse_json(path))
    chunks: list[Chunk] = []
    for doc_id, sec, title, heading, text in records:
        for sub_sec, sub_text in _split_long(sec, text, max_words):
            chunks.append(Chunk(len(chunks), doc_id, sub_sec, title, heading, sub_text,
                                split_sentences(sub_text)))
    if not chunks:
        raise ValueError(f"no documents found under {corpus_dir}")
    return chunks
