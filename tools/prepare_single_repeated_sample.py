#!/usr/bin/env python3
"""Build one exact-length InternEvo sample from consecutive tokenized chunks."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Source InternEvo .bin file")
    parser.add_argument("--output", required=True, help="Destination InternEvo .bin file")
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--num-rows", type=int, default=8)
    parser.add_argument("--expected-tokens", type=int, default=262144)
    return parser.parse_args()


def canonical_token_hash(tokens):
    token_array = np.asarray(tokens, dtype=np.int64)
    return hashlib.sha256(token_array.tobytes(order="C")).hexdigest()


def main():
    args = parse_args()
    source = Path(args.input).resolve()
    source_meta = Path(str(source) + ".meta")
    output = Path(args.output).resolve()
    output_meta = Path(str(output) + ".meta")
    provenance = Path(str(output) + ".provenance.json")

    with source_meta.open("rb") as handle:
        meta = np.load(handle)
    if meta.ndim != 2 or meta.shape[1] < 2:
        raise ValueError(f"unexpected source meta shape: {meta.shape}")

    stop_row = args.start_row + args.num_rows
    if args.start_row < 0 or stop_row > len(meta):
        raise IndexError(f"requested rows [{args.start_row}, {stop_row}) from {len(meta)} rows")

    tokens = []
    row_hashes = []
    with source.open("rb") as handle:
        for row_index in range(args.start_row, stop_row):
            offset = int(meta[row_index, 0])
            expected_length = int(meta[row_index, -1])
            handle.seek(offset)
            raw_line = handle.readline()
            record = json.loads(raw_line.decode("utf-8"))
            row_tokens = record["tokens"]
            if len(row_tokens) != expected_length:
                raise ValueError(
                    f"row {row_index}: JSON length {len(row_tokens)} != meta length {expected_length}"
                )
            tokens.extend(row_tokens)
            row_hashes.append(hashlib.sha256(raw_line).hexdigest())

    if len(tokens) != args.expected_tokens:
        raise ValueError(f"derived token count {len(tokens)} != expected {args.expected_tokens}")
    if min(tokens) < 0 or max(tokens) >= 92544:
        raise ValueError(f"token IDs out of range: min={min(tokens)}, max={max(tokens)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps({"tokens": tokens}, separators=(",", ":")) + "\n").encode("utf-8")
    output.write_bytes(encoded)
    with output_meta.open("wb") as handle:
        np.save(handle, np.asarray([[0, len(tokens)]], dtype=np.int64))

    details = {
        "source_bin": str(source),
        "source_bin_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_rows": list(range(args.start_row, stop_row)),
        "source_row_sha256": row_hashes,
        "token_count": len(tokens),
        "token_min": min(tokens),
        "token_max": max(tokens),
        "tokens_int64_sha256": canonical_token_hash(tokens),
        "output_bin": str(output),
        "output_bin_sha256": hashlib.sha256(encoded).hexdigest(),
        "meta_shape": [1, 2],
        "meta": [[0, len(tokens)]],
    }
    provenance.write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(details, indent=2))


if __name__ == "__main__":
    main()
