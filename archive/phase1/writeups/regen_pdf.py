#!/usr/bin/env python3
"""Regenerate writeups/phase1_algo_contribution.{html,pdf} from the markdown source."""

from __future__ import annotations

import html
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "writeups" / "phase1_algo_contribution.md"
HTML = ROOT / "writeups" / "phase1_algo_contribution.html"
PDF = ROOT / "writeups" / "phase1_algo_contribution.pdf"
DIST = ROOT / "writeups" / "dist"


def inline(s: str) -> str:
    parts: list[str] = []
    last = 0
    for m in re.finditer(r"`([^`]+)`", s):
        parts.append(html.escape(s[last : m.start()]))
        parts.append(f"<code>{html.escape(m.group(1))}</code>")
        last = m.end()
    parts.append(html.escape(s[last:]))
    s2 = "".join(parts)
    s2 = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s2)
    s2 = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", s2)
    s2 = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s2)
    return s2


def convert(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    in_code = False
    table_rows: list[str] = []

    def flush_table() -> None:
        nonlocal table_rows
        if not table_rows:
            return
        out.append('<table border="1" cellpadding="4" cellspacing="0">')
        for ri, row in enumerate(table_rows):
            cells = [c.strip() for c in row.strip("|").split("|")]
            if all(set(c) <= set("-: ") and c for c in cells):
                continue
            tag = "th" if ri == 0 else "td"
            out.append(
                "<tr>" + "".join(f"<{tag}>{inline(c)}</{tag}>" for c in cells) + "</tr>"
            )
        out.append("</table>")
        table_rows = []

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            if table_rows:
                flush_table()
            if not in_code:
                in_code = True
                out.append("<pre><code>")
            else:
                in_code = False
                out.append("</code></pre>")
            i += 1
            continue
        if in_code:
            out.append(html.escape(line) + "\n")
            i += 1
            continue
        if line.strip().startswith("|"):
            table_rows.append(line)
            i += 1
            continue
        if table_rows:
            flush_table()

        if line.strip() == "---":
            out.append("<hr/>")
        elif line.startswith("# "):
            out.append(f"<h1>{inline(line[2:])}</h1>")
        elif line.startswith("## "):
            out.append(f"<h2>{inline(line[3:])}</h2>")
        elif line.startswith("### "):
            out.append(f"<h3>{inline(line[4:])}</h3>")
        elif line.strip() == "":
            out.append("<p></p>")
        elif line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(f"<li>{inline(lines[i][2:])}</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        elif re.match(r"^\d+\. ", line):
            items = []
            while i < len(lines) and re.match(r"^\d+\. ", lines[i]):
                body = re.sub(r"^\d+\. ", "", lines[i])
                items.append(f"<li>{inline(body)}</li>")
                i += 1
            out.append("<ol>" + "".join(items) + "</ol>")
            continue
        elif line.strip() == r"\[":
            block = [line]
            i += 1
            while i < len(lines) and lines[i].strip() != r"\]":
                block.append(lines[i])
                i += 1
            if i < len(lines):
                block.append(lines[i])
            out.append("<p>" + "<br/>".join(html.escape(b) for b in block) + "</p>")
        else:
            out.append(f"<p>{inline(line)}</p>")
        i += 1
    if table_rows:
        flush_table()

    css = (
        "body{font-family:Georgia,serif;max-width:820px;margin:2rem auto;"
        "line-height:1.45;font-size:11pt}"
        "table{border-collapse:collapse;margin:1em 0;font-size:10pt}"
        "th,td{border:1px solid #444;padding:4px 6px;vertical-align:top}"
        "code,pre{font-family:Menlo,monospace;font-size:9.5pt}"
        "pre{background:#f6f6f6;padding:0.8em;overflow:auto}"
        "h1{font-size:1.6rem}h2{font-size:1.25rem;margin-top:1.6em}"
    )
    return (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
        "<title>Phase 1 Algorithmic Contribution</title>"
        f"<style>{css}</style></head><body>\n"
        + "\n".join(out)
        + "\n</body></html>\n"
    )


def main() -> None:
    HTML.write_text(convert(MD.read_text()))
    print(f"wrote {HTML} ({HTML.stat().st_size} bytes)")
    cmd = [
        "google-chrome",
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF.resolve()}",
        HTML.resolve().as_uri(),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"chrome failed: {r.stderr}")
    print(f"wrote {PDF} ({PDF.stat().st_size} bytes)")
    DIST.mkdir(exist_ok=True)
    for src in (PDF, ROOT / "writeups" / "PHASE1_NOMINATIONS.md", ROOT / "writeups" / "artifacts" / "bias_var_mini20_v010.json"):
        (DIST / src.name).write_bytes(src.read_bytes())
    print(f"refreshed {DIST}")


if __name__ == "__main__":
    main()
