"""Inspect public corpus metadata/NPZ headers without downloading large shards.

Offline, read-only research tool; never imported by an estimator.
"""

import argparse
import ast
import concurrent.futures
import hashlib
import json
import struct
import urllib.error
import urllib.request
import zlib
from datetime import datetime, timezone
from pathlib import Path

REPO = "keenanpepper/arc-whestbench-p2-d8b-corpus-14k"
REVISION = "cc3405a9393bc398e0af4b0e5d6e78da0ad5f6c8"
ROOT = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fetch(url):
    with urllib.request.urlopen(url, timeout=45) as response:
        return response.read()


def region(url, start, stop):
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{stop}"})
    with urllib.request.urlopen(request, timeout=45) as response:
        if response.status != 206:
            raise RuntimeError(f"Range request not respected: {response.status}")
        content_range = response.headers.get("Content-Range")
        if not content_range.startswith(f"bytes {start}-{stop}/"):
            raise RuntimeError(f"Unexpected Content-Range: {content_range}")
        data = response.read(stop - start + 2)
    if len(data) != stop - start + 1:
        raise RuntimeError("Range payload length mismatch")
    return data


def field_header(url, entry):
    offset = entry["local_header_offset"]
    head = region(url, offset, offset + 29)
    if head[:4] != b"PK\x03\x04":
        raise RuntimeError("Bad local ZIP header")
    name_size, extra_size = struct.unpack_from("<HH", head, 26)
    data_offset = offset + 30 + name_size + extra_size
    amount = min(entry["compressed_size"], 8192)
    compressed = region(url, data_offset, data_offset + amount - 1)
    if entry["compression"] == 8:
        decoded = zlib.decompressobj(-15).decompress(compressed, 65536)
    elif entry["compression"] == 0:
        decoded = compressed
    else:
        raise RuntimeError("Unsupported ZIP compression")
    if decoded[:6] != b"\x93NUMPY":
        raise RuntimeError("Not an NPY member")
    major = decoded[6]
    length_size = 2 if major == 1 else 4
    header_size = int.from_bytes(decoded[8 : 8 + length_size], "little")
    header_end = 8 + length_size + header_size
    header = ast.literal_eval(decoded[8 + length_size : header_end].decode("latin1"))
    result = dict(entry, npy_version=list(decoded[6:8]), npy=header)
    # Read only scalar/small completed arrays. Bulk arrays remain untouched.
    if entry["uncompressed_size"] <= 4096 and amount == entry["compressed_size"]:
        result["small_payload_hex"] = decoded[header_end:].hex()
    result["sample_sha256"] = sha(compressed)
    result["bytes_read"] = len(head) + len(compressed)
    return result


def inspect_shard(entry):
    filename, size = entry["path"], entry["size"]
    url = ROOT + filename + "?download=true"
    tail_size = min(size, 65536)
    tail = region(url, size - tail_size, size - 1)
    position = tail.rfind(b"PK\x05\x06")
    if position < 0:
        raise RuntimeError("No ZIP end-of-directory")
    count, directory_size, directory_offset = struct.unpack_from("<HII", tail, position + 10)
    if directory_offset >= size - tail_size:
        central = tail[directory_offset - (size - tail_size) :][:directory_size]
    else:
        central = region(url, directory_offset, directory_offset + directory_size - 1)
    members = []
    position = 0
    for _ in range(count):
        if central[position : position + 4] != b"PK\x01\x02":
            raise RuntimeError("Bad central ZIP header")
        compression = struct.unpack_from("<H", central, position + 10)[0]
        crc, compressed_size, uncompressed_size = struct.unpack_from("<III", central, position + 16)
        name_size, extra_size, comment_size = struct.unpack_from("<HHH", central, position + 28)
        offset = struct.unpack_from("<I", central, position + 42)[0]
        name = central[position + 46 : position + 46 + name_size].decode("utf8")
        members.append(dict(name=name, compression=compression, crc32=f"{crc:08x}",
                            compressed_size=compressed_size, uncompressed_size=uncompressed_size,
                            local_header_offset=offset))
        position += 46 + name_size + extra_size + comment_size
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(field_header, url, member): member for member in members}
        for future in concurrent.futures.as_completed(futures):
            member = futures[future]
            try:
                results.append(future.result())
            except Exception as error:
                results.append(dict(member, error=f"{type(error).__name__}: {error}"))
    return dict(path=filename, canonical_url=ROOT + filename, size_bytes=size,
                lfs=entry.get("lfs"), tail_sha256=sha(tail),
                members=sorted(results, key=lambda row: row["name"]),
                bytes_read=tail_size + sum(row.get("bytes_read", 0) for row in results))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    info_bytes = fetch(f"https://huggingface.co/api/datasets/{REPO}")
    tree_bytes = fetch(f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}?expand=true")
    readme_bytes = fetch(ROOT + "README.md")
    info, tree = json.loads(info_bytes), json.loads(tree_bytes)
    selected = [entry for entry in tree if entry["path"] in
                {"shard00.npz", "teacher_shard00.npz", "shard_ext00.npz"}]
    result = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), repository=REPO,
                  frozen_revision=REVISION, source_script_sha256=sha(Path(__file__).read_bytes()),
                  metadata=dict(commit=info.get("sha"), last_modified=info.get("lastModified"),
                                private=info.get("private"), gated=info.get("gated"),
                                card_data=info.get("cardData")),
                  tree=tree, readme=readme_bytes.decode(),
                  source_sha256=dict(info=sha(info_bytes), tree=sha(tree_bytes), readme=sha(readme_bytes)),
                  files=[])
    for entry in selected:
        print("Inspecting", entry["path"], flush=True)
        try:
            result["files"].append(inspect_shard(entry))
        except Exception as error:
            result["files"].append(dict(path=entry["path"], error=f"{type(error).__name__}: {error}"))
        print("Completed", entry["path"], flush=True)
    result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output, flush=True)


if __name__ == "__main__":
    main()
