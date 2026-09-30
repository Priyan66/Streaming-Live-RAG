"""Renders the event log as a self-contained HTML branch timeline (demo aid).

One panel per turn: transcript chunks and utterance end on a stream-time axis,
one lane per speculative branch (forked -> pruned / promoted), retrieval and
answer markers, then the answer with citations and the claim table.
"""
from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path

W, LANE, LEFT, TOP = 760, 26, 150, 28
COLORS = {"promoted": "var(--good)", "pruned": "var(--muted)", "alive": "var(--accent)"}


def _x(t: float, tmax: float) -> float:
    return LEFT + (W - LEFT - 20) * (t / tmax if tmax else 0)


def _turn_svg(evs: list[dict]) -> str:
    chunks = [e for e in evs if e["event"] == "chunk_received"]
    end = next((e for e in evs if e["event"] == "utterance_end"), None)
    ready = next((e for e in evs if e["event"] == "turn_summary"), None)
    tmax = max([e["t_stream"] for e in evs if e.get("t_stream") is not None] + [0.5]) * 1.08
    branches: dict[str, dict] = {}
    for e in evs:
        b = e.get("branch")
        if e["event"] == "branch_forked":
            branches[b] = {"start": e["t_stream"], "end": None, "status": "alive", "label": e.get("cluster_label", ""),
                           "trigger": e.get("trigger", ""), "hyp": e.get("hypothesis", ""), "retr": []}
        elif b in branches and e["event"] == "branch_pruned":
            branches[b].update(end=e["t_stream"], status="pruned", why=e.get("reason", ""))
        elif b in branches and e["event"] == "branch_promoted":
            branches[b].update(end=e["t_stream"], status="promoted")
        elif b in branches and e["event"] == "retrieval_started":
            branches[b]["retr"].append(e["t_stream"])
    lanes = list(branches.items())
    other_retr = [e for e in evs if e["event"] == "retrieval_started" and e.get("branch") not in branches]
    h = TOP + LANE * (len(lanes) + 2) + 24
    out = [f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="branch timeline">']
    for i in range(0, int(tmax * 2) + 1):  # 0.5 s grid
        x = _x(i / 2, tmax)
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{TOP - 8}" y2="{h - 18}" class="grid"/>'
                   f'<text x="{x:.1f}" y="{h - 4}" class="tick">{i / 2:.1f}s</text>')
    y = TOP
    out.append(f'<text x="8" y="{y + 5}" class="lab">transcript</text>')
    for c in chunks:
        x = _x(c["t_stream"], tmax)
        out.append(f'<circle cx="{x:.1f}" cy="{y}" r="5" class="chunk"><title>{html.escape(c.get("text", ""))}'
                   f'</title></circle>')
    if end:
        x = _x(end["t_stream"], tmax)
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{TOP - 12}" y2="{h - 18}" class="endline"/>'
                   f'<text x="{x + 4:.1f}" y="{TOP - 14}" class="tick">utterance end</text>')
    if ready:
        x = _x(ready["t_stream"], tmax)
        out.append(f'<path d="M{x:.1f},{TOP - 6} l6,6 l-6,6 l-6,-6z" class="ready"><title>answer ready '
                   f'{ready["t_stream"]:.3f}s</title></path>')
    for i, (bid, b) in enumerate(lanes):
        y = TOP + LANE * (i + 1)
        x0 = _x(b["start"], tmax)
        x1 = _x(b["end"] if b["end"] is not None else tmax / 1.08, tmax)
        out.append(f'<text x="8" y="{y + 4}" class="lab">{html.escape(b["trigger"])} · {html.escape(b["label"][:16])}'
                   f'</text><rect x="{x0:.1f}" y="{y - 6}" width="{max(3, x1 - x0):.1f}" height="12" rx="6" '
                   f'fill="{COLORS[b["status"]]}"><title>{html.escape(bid)} {b["status"]}: '
                   f'{html.escape(b["hyp"])} {html.escape(b.get("why", ""))}</title></rect>')
        for t in b["retr"]:
            out.append(f'<circle cx="{_x(t, tmax):.1f}" cy="{y}" r="3" class="retr"/>')
    y = TOP + LANE * (len(lanes) + 1)
    if other_retr:
        out.append(f'<text x="8" y="{y + 4}" class="lab">final / patch</text>')
        for e in other_retr:
            out.append(f'<circle cx="{_x(e["t_stream"], tmax):.1f}" cy="{y}" r="4" class="retr2"><title>'
                       f'{html.escape(e.get("trigger", ""))}: {html.escape(e.get("query", ""))}</title></circle>')
    out.append("</svg>")
    return "".join(out)


def write_dashboard(events: list[dict], path: str | Path):
    turns: dict[tuple, list[dict]] = defaultdict(list)
    for e in events:
        if e.get("turn"):
            turns[(e["session_id"], e["turn"])].append(e)
    panels = []
    for (sid, turn), evs in turns.items():
        summ = next((e for e in evs if e["event"] == "turn_summary"), {})
        av = next((e for e in evs if e["event"] == "answer_version"), None)
        end = next((e for e in evs if e["event"] == "utterance_end"), {})
        lat = summ.get("latency", {})
        rows = "".join(
            f"<tr><td>{html.escape(str(c.get('id')))}</td><td>v{c.get('version')}</td><td>{html.escape(str(c.get('status')))}"
            f"</td><td>{html.escape(str(c.get('citation')))}</td></tr>" for c in (av or {}).get("claims", []))
        answer = html.escape((av or {}).get("answer", "")).replace("\n", "<br>")
        unc = (av or {}).get("uncertainty")
        panels.append(f"""
<section class="card">
  <header><span class="pill">{html.escape(summ.get('kind', '?'))}</span>
    <b>session {sid} · turn {turn}</b>
    <span class="meta">v{summ.get('answer_version', '-')} · first retrieval {lat.get('first_retrieval_s')}s ·
    end {lat.get('utterance_end_s')}s · +{lat.get('post_utterance_ms')} ms</span></header>
  <p class="transcript">“{html.escape(end.get('transcript', ''))}”</p>
  {_turn_svg(evs)}
  <p class="answer">{answer}</p>
  {f'<p class="unc">⚠ {html.escape(unc)}</p>' if unc else ''}
  {f'<table><tr><th>claim</th><th>ver</th><th>status</th><th>citation</th></tr>{rows}</table>' if rows else ''}
</section>""")
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Streaming RAG Trace</title>
<style>
:root{{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1f;--muted:#9a9a9f;--line:#e4e4e7;--accent:#3b6fd8;--good:#2f9e6b;--warn:#b7791f}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#121214;--card:#1c1c1f;--ink:#ececef;--muted:#6b6b72;--line:#2c2c31;--accent:#6f9bff;--good:#46c28e;--warn:#e0a84a}}}}
:root[data-theme="dark"]{{--bg:#121214;--card:#1c1c1f;--ink:#ececef;--muted:#6b6b72;--line:#2c2c31;--accent:#6f9bff;--good:#46c28e;--warn:#e0a84a}}
body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,sans-serif;padding:24px 16px}}
main{{max-width:820px;margin:auto}} h1{{font-size:20px;margin:0 0 4px}} .sub{{color:var(--muted);margin:0 0 18px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:0 0 16px}}
header{{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}} .meta{{color:var(--muted);font-size:12px}}
.pill{{background:var(--accent);color:#fff;border-radius:999px;padding:1px 9px;font-size:12px}}
.transcript{{font-style:italic;margin:8px 0}} .answer{{white-space:normal}} .unc{{color:var(--warn)}}
svg{{width:100%;height:auto;display:block}} .grid{{stroke:var(--line)}} .tick{{fill:var(--muted);font-size:10px}}
.lab{{fill:var(--ink);font-size:11px}} .chunk{{fill:var(--accent)}} .endline{{stroke:var(--warn);stroke-dasharray:4 3}}
.ready{{fill:var(--good)}} .retr{{fill:var(--card);stroke:var(--ink);stroke-width:1.2}} .retr2{{fill:var(--warn)}}
table{{border-collapse:collapse;width:100%;font-size:12px}} td,th{{border-top:1px solid var(--line);padding:3px 6px;text-align:left}}
</style></head><body><main><h1>Streaming RAG trace</h1>
<p class="sub">Branch lanes: blue = alive, green = promoted, grey = pruned; hollow dots = branch retrievals, amber dots = final/patch retrievals; diamond = answer ready.</p>
{''.join(panels)}
<details><summary>raw events ({len(events)})</summary><pre style="white-space:pre-wrap;font-size:11px">{html.escape(json.dumps(events[:400], indent=1, ensure_ascii=False))}</pre></details>
</main></body></html>"""
    Path(path).write_text(doc, encoding="utf-8")
