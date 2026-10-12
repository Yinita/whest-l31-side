"""Download linked public reports and extract text, preserve failure provenance."""

import hashlib
import json
import subprocess
from pathlib import Path

import requests


def main():
    entries = json.load(open("results/phase2/discussion_attachment_inventory_20261012.json"))
    root = Path("benchmark/data/discussion_20261012/pdfs")
    root.mkdir(exist_ok=True)
    for i, e in enumerate(entries):
        p = root / f"{e['topic_id']}_{i}.pdf"
        try:
            r = requests.get(e["url"], timeout=45)
            r.raise_for_status()
            if not r.content.startswith(b"%PDF"):
                raise ValueError("Not PDF response")
            p.write_bytes(r.content)
            subprocess.run(
                ["pdftotext", "-layout", str(p), str(p.with_suffix(".txt"))],
                check=True,
                capture_output=True,
            )
            e.update(
                status="text_extracted_not_yet_full_analysis",
                sha256=hashlib.sha256(r.content).hexdigest(),
                path=str(p),
                text_path=str(p.with_suffix(".txt")),
            )
            print(i, e["topic_id"], "ok", flush=True)
        except Exception as exc:
            e.update(status="failed", error=str(exc))
            print(i, "failed", flush=True)
    Path("results/phase2/discussion_pdf_manifest_20261012.json").write_text(
        json.dumps(entries, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
