#!/usr/bin/env python3
"""Rechunk InternEvo JSONL token files into exact fixed-length documents."""

import argparse
import json
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--document-tokens", type=int, required=True)
    parser.add_argument("--skip-output-documents", type=int, default=0)
    parser.add_argument("--max-documents", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.document_tokens <= 0:
        raise ValueError("--document-tokens must be positive")

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)

    offsets = []
    pending = []
    byte_offset = 0
    input_documents = 0
    input_tokens = 0
    complete_documents_seen = 0

    with open(output, "wb") as target:
        for input_path in args.input:
            with open(input_path, "rb") as source:
                for line in source:
                    if not line.strip():
                        continue
                    tokens = json.loads(line)["tokens"]
                    input_documents += 1
                    input_tokens += len(tokens)
                    pending.extend(tokens)

                    while len(pending) >= args.document_tokens:
                        document = pending[: args.document_tokens]
                        del pending[: args.document_tokens]
                        complete_documents_seen += 1
                        if complete_documents_seen <= args.skip_output_documents:
                            continue
                        encoded = (
                            json.dumps({"tokens": document}, separators=(",", ":")) + "\n"
                        ).encode("utf-8")
                        target.write(encoded)
                        offsets.append((byte_offset, args.document_tokens))
                        byte_offset += len(encoded)

                        if args.max_documents and len(offsets) >= args.max_documents:
                            pending.clear()
                            break
                    if args.max_documents and len(offsets) >= args.max_documents:
                        break
            if args.max_documents and len(offsets) >= args.max_documents:
                break

    with open(output + ".meta", "wb") as meta_file:
        np.save(meta_file, np.asarray(offsets, dtype=np.int64))

    print(
        json.dumps(
            {
                "input_documents": input_documents,
                "input_tokens_read": input_tokens,
                "output_documents": len(offsets),
                "output_tokens": len(offsets) * args.document_tokens,
                "skipped_output_documents": min(
                    complete_documents_seen, args.skip_output_documents
                ),
                "discarded_tail_tokens": len(pending),
                "document_tokens": args.document_tokens,
                "output": output,
            }
        )
    )


if __name__ == "__main__":
    main()
