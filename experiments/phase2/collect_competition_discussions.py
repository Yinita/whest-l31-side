"""Archive all accessible category topics and replies for research provenance."""

import datetime
import json
import time
from pathlib import Path

import requests


def main():
    root = Path("benchmark/data/discussion_20261012")
    root.mkdir(exist_ok=True)
    session = requests.Session()
    topics = {}
    page = 0
    while True:
        response = session.get(
            f"https://discourse.aicrowd.com/c/white-box-estimation-challenge-2026/2991.json?page={page}",
            timeout=40,
        )
        response.raise_for_status()
        data = response.json()
        root.joinpath(f"category_{page}.json").write_text(json.dumps(data))
        batch = data.get("topic_list", {}).get("topics", [])
        for topic in batch:
            topics[topic["id"]] = topic
        if not data.get("topic_list", {}).get("more_topics_url") or not batch:
            break
        page += 1
    failures = []
    index = []
    for tid, topic in topics.items():
        try:
            response = session.get(f"https://discourse.aicrowd.com/t/{tid}.json", timeout=40)
            response.raise_for_status()
            data = response.json()
            stream = data.get("post_stream", {})
            posts = stream.get("posts", [])
            missing = set(stream.get("stream", [])) - {p["id"] for p in posts}
            for offset in range(0, len(missing), 20):
                params = [("post_ids[]", p) for p in sorted(missing)[offset : offset + 20]]
                r = session.get(
                    f"https://discourse.aicrowd.com/t/{tid}/posts.json", params=params, timeout=40
                )
                r.raise_for_status()
                posts += r.json().get("post_stream", {}).get("posts", [])
            data["post_stream"]["posts"] = posts
            root.joinpath(f"topic_{tid}.json").write_text(json.dumps(data))
            index.append(
                dict(
                    id=tid,
                    title=topic["title"],
                    url=f"https://discourse.aicrowd.com/t/{topic['slug']}/{tid}",
                    posts=len(posts),
                    expected_posts=len(stream.get("stream", [])),
                )
            )
            print(tid, len(posts), flush=True)
        except Exception as e:
            failures.append(dict(id=tid, error=str(e)))
        time.sleep(0.2)
    out = Path("results/phase2/discussion_inventory_20261012.json")
    out.write_text(
        json.dumps(
            dict(
                collected_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                topics=index,
                failures=failures,
                category_pages=page + 1,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print("TOTAL", len(index), "FAILURES", len(failures), flush=True)


if __name__ == "__main__":
    main()
