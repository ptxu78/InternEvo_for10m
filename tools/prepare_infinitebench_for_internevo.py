#!/usr/bin/env python3
"""Convert InfiniteBench JSONL records to InternEvo tokenized .bin/.meta files."""

import argparse
import json
import os
import sys

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--transformers-dir", required=True)
    return parser.parse_args()


def render_record(record):
    parts = []
    context = record.get("context", "")
    question = record.get("input", "")
    options = record.get("options") or []
    answers = record.get("answer") or []
    if isinstance(answers, str):
        answers = [answers]

    if context:
        parts.append("Context:\n" + context)
    if question:
        parts.append("Question:\n" + question)
    if options:
        parts.append("Options:\n" + "\n".join(f"{i + 1}. {option}" for i, option in enumerate(options)))
    if answers:
        parts.append("Answer:\n" + "\n".join(answers))
    return "\n\n".join(parts)


def main():
    args = parse_args()
    sys.path.insert(0, os.path.abspath(args.transformers_dir))
    from internlm_model import InternLMTokenizer

    tokenizer = InternLMTokenizer(vocab_file=args.tokenizer_model, add_bos_token=True, add_eos_token=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    offsets = []
    offset = 0
    token_count = 0
    with open(args.input, "r", encoding="utf-8") as source, open(args.output, "wb") as target:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            text = render_record(record)
            tokens = tokenizer.encode(text)
            encoded = (json.dumps({"tokens": tokens}, separators=(",", ":")) + "\n").encode("utf-8")
            offsets.append((offset, len(tokens)))
            offset += len(encoded)
            token_count += len(tokens)
            target.write(encoded)

    meta = np.asarray(offsets, dtype=np.int64)
    with open(args.output + ".meta", "wb") as meta_file:
        np.save(meta_file, meta)
    print(json.dumps({"documents": len(offsets), "tokens": token_count, "output": args.output}))


if __name__ == "__main__":
    main()
