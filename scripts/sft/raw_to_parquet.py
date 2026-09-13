import argparse
import json
import os
import random
from collections import Counter

import pandas as pd

from search_r1.sft.trajectory_utils import build_parquet_rows, split_train_val_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Convert teacher rollout JSONL into Search-R1 SFT parquet files.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--train-size", type=int, default=None,
                        help="Exact train size after filtering/deduplication; all remaining rows become validation.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--require-quality-pass", action="store_true")
    parser.add_argument("--dedupe-key", default="question")
    return parser.parse_args()


def load_records(input_jsonl):
    records = []
    with open(input_jsonl, "r", encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_summary(output_dir, train_rows, val_rows):
    counts = Counter()
    for row in train_rows:
        counts[f"train::{row['data_source']}"] += 1
    for row in val_rows:
        counts[f"val::{row['data_source']}"] += 1

    summary = {
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "counts": dict(counts),
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as fout:
        json.dump(summary, fout, ensure_ascii=False, indent=2)


def rows_to_dataframe(rows, fallback_rows):
    if rows:
        return pd.DataFrame(rows)
    if fallback_rows:
        return pd.DataFrame(columns=pd.DataFrame(fallback_rows).columns)
    return pd.DataFrame(columns=["prompt", "response", "data_source", "ground_truth", "extra_info"])


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    records = load_records(args.input_jsonl)
    rows = build_parquet_rows(
        records=records,
        require_quality_pass=args.require_quality_pass,
        dedupe_key=args.dedupe_key,
    )
    if args.train_size is not None:
        if not 0 < args.train_size < len(rows):
            raise ValueError(f"train-size must leave nonempty train and val splits; got {args.train_size} of {len(rows)}")
        random.Random(args.seed).shuffle(rows)
        train_rows, val_rows = rows[:args.train_size], rows[args.train_size:]
    else:
        train_rows, val_rows = split_train_val_rows(rows, val_ratio=args.val_ratio, seed=args.seed)

    rows_to_dataframe(train_rows, val_rows).to_parquet(os.path.join(args.output_dir, "train.parquet"))
    rows_to_dataframe(val_rows, train_rows).to_parquet(os.path.join(args.output_dir, "val.parquet"))
    write_summary(args.output_dir, train_rows, val_rows)


if __name__ == "__main__":
    main()
