#!/usr/bin/env python3
"""Build fixed-prefix, varying-tail InternEvo samples from tokenized rows."""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Source InternEvo .bin file")
    parser.add_argument("--output", required=True, help="Destination InternEvo .bin file")
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--prefix-blocks", type=int, default=9)
    parser.add_argument("--tail-blocks", type=int, default=1)
    parser.add_argument("--rows-per-block", type=int, default=32)
    parser.add_argument("--tokens-per-row", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def main():
    args = parse_args()
    source = Path(args.input).resolve()
    source_meta = Path(str(source) + ".meta")
    output = Path(args.output).resolve()
    output_meta = Path(str(output) + ".meta")
    provenance = Path(str(output) + ".provenance.json")
    temp_output = Path(str(output) + ".tmp")

    prefix_rows = args.prefix_blocks * args.rows_per_block
    tail_rows = args.tail_blocks * args.rows_per_block
    prefix_tokens = prefix_rows * args.tokens_per_row
    tail_tokens = tail_rows * args.tokens_per_row
    sample_tokens = prefix_tokens + tail_tokens

    with source_meta.open("rb") as handle:
        meta = np.load(handle)
    if meta.ndim != 2 or meta.shape[1] < 2:
        raise ValueError(f"unexpected source meta shape: {meta.shape}")
    if not np.all(meta[:, -1] == args.tokens_per_row):
        unique_lengths = np.unique(meta[:, -1]).tolist()
        raise ValueError(f"source row lengths are not uniform: {unique_lengths}")

    first_tail_start = prefix_rows
    last_tail_start = len(meta) - tail_rows
    candidates = list(range(first_tail_start, last_tail_start + 1))
    if args.num_samples > len(candidates):
        raise ValueError(
            f"requested {args.num_samples} tails, but only {len(candidates)} "
            "unique contiguous windows are available"
        )

    rng = random.Random(args.seed)
    other_starts = candidates[1:]
    rng.shuffle(other_starts)
    tail_starts = [first_tail_start] + other_starts[: args.num_samples - 1]

    targets = [output, output_meta, provenance, temp_output]
    existing = [str(path) for path in targets if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"refusing to overwrite: {existing}")
    for path in targets:
        if path.exists():
            path.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)

    row_payloads = []
    row_hashes = []
    token_min = None
    token_max = None
    with source.open("rb") as handle:
        for row_index, row_meta in enumerate(meta):
            handle.seek(int(row_meta[0]))
            raw_line = handle.readline()
            record = json.loads(raw_line.decode("utf-8"))
            tokens = record["tokens"]
            if len(tokens) != args.tokens_per_row:
                raise ValueError(
                    f"row {row_index}: JSON length {len(tokens)} != {args.tokens_per_row}"
                )
            row_min = min(tokens)
            row_max = max(tokens)
            token_min = row_min if token_min is None else min(token_min, row_min)
            token_max = row_max if token_max is None else max(token_max, row_max)
            payload = ",".join(str(token) for token in tokens).encode("ascii")
            row_payloads.append(payload)
            row_hashes.append(hashlib.sha256(raw_line).hexdigest())
            if (row_index + 1) % 100 == 0 or row_index + 1 == len(meta):
                print(f"prepared source rows: {row_index + 1}/{len(meta)}", flush=True)

    if token_min < 0 or token_max >= 92544:
        raise ValueError(f"token IDs out of range: min={token_min}, max={token_max}")

    fixed_payload = b",".join(row_payloads[:prefix_rows])
    line_prefix = b'{"tokens":['
    line_suffix = b"]}\n"
    offsets = []
    output_digest = hashlib.sha256()
    tail_details = []

    with temp_output.open("wb") as handle:
        for sample_index, tail_start in enumerate(tail_starts):
            tail_end = tail_start + tail_rows
            tail_payload = b",".join(row_payloads[tail_start:tail_end])
            offsets.append([handle.tell(), sample_tokens])
            chunks = (line_prefix, fixed_payload, b",", tail_payload, line_suffix)
            for chunk in chunks:
                handle.write(chunk)
                output_digest.update(chunk)
            tail_details.append(
                {
                    "sample_index": sample_index,
                    "source_rows": [tail_start, tail_end],
                    "payload_sha256": hashlib.sha256(tail_payload).hexdigest(),
                }
            )
            if (sample_index + 1) % 10 == 0 or sample_index + 1 == args.num_samples:
                gib = handle.tell() / (1024**3)
                print(
                    f"wrote samples: {sample_index + 1}/{args.num_samples} ({gib:.2f} GiB)",
                    flush=True,
                )
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(temp_output, output)
    with output_meta.open("wb") as handle:
        np.save(handle, np.asarray(offsets, dtype=np.int64))

    fixed_blocks = []
    for block_index in range(args.prefix_blocks):
        start = block_index * args.rows_per_block
        end = start + args.rows_per_block
        fixed_blocks.append(
            {
                "block_index": block_index,
                "source_rows": [start, end],
                "token_count": args.rows_per_block * args.tokens_per_row,
            }
        )

    details = {
        "source_bin": str(source),
        "source_bin_sha256": sha256_file(source),
        "source_meta": str(source_meta),
        "source_rows": int(len(meta)),
        "source_tokens": int(meta[:, -1].sum()),
        "seed": args.seed,
        "num_samples": args.num_samples,
        "sample_tokens": sample_tokens,
        "prefix_blocks": args.prefix_blocks,
        "tail_blocks": args.tail_blocks,
        "prefix_tokens": prefix_tokens,
        "tail_tokens": tail_tokens,
        "fixed_prefix_blocks": fixed_blocks,
        "tail_windows": tail_details,
        "tail_windows_unique": len({item["payload_sha256"] for item in tail_details}),
        "token_min": token_min,
        "token_max": token_max,
        "source_row_sha256": row_hashes,
        "output_bin": str(output),
        "output_bytes": output.stat().st_size,
        "output_bin_sha256": output_digest.hexdigest(),
        "meta_shape": [args.num_samples, 2],
    }
    provenance.write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in details.items() if key not in {"source_row_sha256", "tail_windows"}}, indent=2))


if __name__ == "__main__":
    main()
