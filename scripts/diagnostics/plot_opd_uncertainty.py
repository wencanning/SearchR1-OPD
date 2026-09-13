#!/usr/bin/env python3
import argparse
import gzip
import html
import json
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


def _write_svg_bar(grouped: pd.DataFrame, x: str, y: str, path: Path, title: str):
    width, height = 900, 420
    margin_left, margin_bottom, margin_top = 70, 110, 40
    plot_width = width - margin_left - 20
    plot_height = height - margin_top - margin_bottom
    min_y = min(float(grouped["mean"].min()), 0.0)
    max_y = max(float(grouped["mean"].max()), 0.0)
    if max_y == min_y:
        max_y = min_y + 1.0
    bar_width = max(plot_width / max(len(grouped), 1) * 0.75, 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="{width/2}" y="24" text-anchor="middle" font-size="18">{html.escape(title)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height-margin_bottom}" stroke="black"/>',
        f'<line x1="{margin_left}" y1="{height-margin_bottom}" x2="{width-20}" y2="{height-margin_bottom}" stroke="black"/>',
    ]
    zero_y = height - margin_bottom - (0.0 - min_y) / (max_y - min_y) * plot_height
    parts.append(f'<line x1="{margin_left}" y1="{zero_y:.1f}" x2="{width-20}" y2="{zero_y:.1f}" stroke="#999" stroke-dasharray="3,3"/>')
    for i, row in grouped.reset_index(drop=True).iterrows():
        x_pos = margin_left + (i + 0.5) * plot_width / max(len(grouped), 1) - bar_width / 2
        value = float(row["mean"])
        value_y = height - margin_bottom - (value - min_y) / (max_y - min_y) * plot_height
        y_pos = min(value_y, zero_y)
        bar_h = abs(value_y - zero_y)
        label = html.escape(str(row[x]))
        parts.append(f'<rect x="{x_pos:.1f}" y="{y_pos:.1f}" width="{bar_width:.1f}" height="{bar_h:.1f}" fill="#4C78A8"/>')
        parts.append(
            f'<text x="{x_pos + bar_width/2:.1f}" y="{height-margin_bottom+14}" '
            f'text-anchor="end" transform="rotate(-35 {x_pos + bar_width/2:.1f},{height-margin_bottom+14})" '
            f'font-size="11">{label}</text>'
        )
    parts.append(f'<text x="12" y="{margin_top+10}" font-size="12">range {y}: {min_y:.3f}-{max_y:.3f}</text>')
    parts.append("</svg>")
    path.with_suffix(".svg").write_text("\n".join(parts), encoding="utf-8")


def _write_svg_line(grouped: pd.DataFrame, x: str, y: str, hue: Optional[str], path: Path, title: str):
    width, height = 900, 420
    margin_left, margin_bottom, margin_top = 70, 70, 40
    plot_width = width - margin_left - 30
    plot_height = height - margin_top - margin_bottom
    x_vals = sorted(grouped[x].dropna().unique())
    if not x_vals:
        return
    min_x, max_x = float(min(x_vals)), float(max(x_vals))
    min_y, max_y = float(grouped[y].min()), float(grouped[y].max())
    if max_y == min_y:
        max_y = min_y + 1.0
    if max_x == min_x:
        max_x = min_x + 1.0

    def sx(value):
        return margin_left + (float(value) - min_x) / (max_x - min_x) * plot_width

    def sy(value):
        return height - margin_bottom - (float(value) - min_y) / (max_y - min_y) * plot_height

    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="{width/2}" y="24" text-anchor="middle" font-size="18">{html.escape(title)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height-margin_bottom}" stroke="black"/>',
        f'<line x1="{margin_left}" y1="{height-margin_bottom}" x2="{width-30}" y2="{height-margin_bottom}" stroke="black"/>',
        f'<text x="12" y="{margin_top+10}" font-size="12">range {y}: {min_y:.3f}-{max_y:.3f}</text>',
    ]
    group_iter = grouped.groupby(hue, dropna=False) if hue and hue in grouped.columns else [("mean", grouped)]
    for color_idx, (label, group) in enumerate(group_iter):
        group = group.sort_values(x)
        points = " ".join(f'{sx(row[x]):.1f},{sy(row[y]):.1f}' for _, row in group.iterrows())
        color = colors[color_idx % len(colors)]
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{width-150}" y="{55 + color_idx * 18}" font-size="12" fill="{color}">{html.escape(str(label))}</text>')
    parts.append("</svg>")
    path.with_suffix(".svg").write_text("\n".join(parts), encoding="utf-8")


def _write_svg_scatter(sample: pd.DataFrame, x: str, y: str, path: Path, title: str):
    if len(sample) > 12000:
        sample = sample.sample(12000, random_state=0)
    width, height = 700, 520
    margin_left, margin_bottom, margin_top = 70, 60, 40
    plot_width = width - margin_left - 30
    plot_height = height - margin_top - margin_bottom
    min_x, max_x = float(sample[x].min()), float(sample[x].max())
    min_y, max_y = float(sample[y].min()), float(sample[y].max())
    if max_x == min_x:
        max_x = min_x + 1.0
    if max_y == min_y:
        max_y = min_y + 1.0

    def sx(value):
        return margin_left + (float(value) - min_x) / (max_x - min_x) * plot_width

    def sy(value):
        return height - margin_bottom - (float(value) - min_y) / (max_y - min_y) * plot_height

    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]
    segments = {segment: colors[i % len(colors)] for i, segment in enumerate(sorted(sample["segment"].astype(str).unique()))}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="{width/2}" y="24" text-anchor="middle" font-size="18">{html.escape(title)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height-margin_bottom}" stroke="black"/>',
        f'<line x1="{margin_left}" y1="{height-margin_bottom}" x2="{width-30}" y2="{height-margin_bottom}" stroke="black"/>',
    ]
    for _, row in sample.iterrows():
        color = segments[str(row["segment"])]
        parts.append(f'<circle cx="{sx(row[x]):.1f}" cy="{sy(row[y]):.1f}" r="1.6" fill="{color}" opacity="0.35"/>')
    parts.append("</svg>")
    path.with_suffix(".svg").write_text("\n".join(parts), encoding="utf-8")


def iter_records(diagnostics_dir: Path, max_files: Optional[int] = None):
    paths = sorted(diagnostics_dir.glob("step_*.jsonl.gz"))
    if max_files is not None:
        paths = paths[:max_files]
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if record.get("record_type") == "trajectory":
                    yield record


def build_tables(diagnostics_dir: Path, max_files: Optional[int] = None):
    token_rows = []
    trajectory_rows = []
    high_entropy_rows = []

    for record in iter_records(diagnostics_dir, max_files=max_files):
        base = {
            "global_step": record["global_step"],
            "epoch": record["epoch"],
            "sequence_index": record["sequence_index"],
            "uid": record.get("uid"),
            "data_source": record.get("data_source"),
            "answer_correct": record.get("answer_correct"),
            "retrieval_hit": record.get("retrieval_hit"),
            "sequence_score": record.get("sequence_score"),
            "sequence_reward": record.get("sequence_reward"),
            "valid_length": record.get("valid_length"),
        }
        trajectory_rows.append({
            **base,
            "final_answer": record.get("final_answer"),
        })

        teacher_entropy = record.get("teacher_entropy") or []
        student_entropy = record.get("student_entropy") or []
        logp_gap = record.get("logp_gap_teacher_minus_student") or []
        segments = record.get("segments") or []
        turn_ids = record.get("turn_ids") or []
        loss_mask = record.get("loss_mask") or []
        token_texts = record.get("token_texts") or [None] * len(teacher_entropy)

        for i, entropy in enumerate(teacher_entropy):
            if i >= len(loss_mask) or not loss_mask[i]:
                continue
            token_rows.append({
                **base,
                "token_index": i,
                "token_text": token_texts[i] if i < len(token_texts) else None,
                "segment": segments[i] if i < len(segments) else "unknown",
                "turn_id": turn_ids[i] if i < len(turn_ids) else None,
                "teacher_entropy": entropy,
                "student_entropy": student_entropy[i] if i < len(student_entropy) else None,
                "entropy_gap_student_minus_teacher": (
                    student_entropy[i] - entropy
                    if i < len(student_entropy) and student_entropy[i] is not None and entropy is not None
                    else None
                ),
                "logp_gap_teacher_minus_student": logp_gap[i] if i < len(logp_gap) else None,
            })

        for token in record.get("top_teacher_entropy_tokens") or []:
            high_entropy_rows.append({**base, **token})

    return (
        pd.DataFrame(token_rows),
        pd.DataFrame(trajectory_rows),
        pd.DataFrame(high_entropy_rows),
    )


def save_bar(df: pd.DataFrame, x: str, y: str, path: Path, title: str):
    grouped = df.groupby(x, dropna=False)[y].agg(["mean", "count", "std"]).reset_index()
    grouped = grouped.sort_values("mean", ascending=False)
    grouped.to_csv(path.with_suffix(".csv"), index=False)

    if plt is None:
        _write_svg_bar(grouped, x, y, path, title)
        return

    plt.figure(figsize=(10, 5))
    plt.bar(grouped[x].astype(str), grouped["mean"], yerr=grouped["std"].fillna(0), capsize=3)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel(y)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_line(df: pd.DataFrame, x: str, y: str, hue: Optional[str], path: Path, title: str):
    if plt is None:
        grouped_cols = [x] + ([hue] if hue and hue in df.columns else [])
        grouped = df.groupby(grouped_cols, dropna=False)[y].mean().reset_index()
        grouped.to_csv(path.with_suffix(".csv"), index=False)
        _write_svg_line(grouped, x, y, hue, path, title)
        return

    plt.figure(figsize=(10, 5))
    if hue and hue in df.columns and df[hue].notna().any():
        for label, group in df.groupby(hue):
            grouped = group.groupby(x)[y].mean().reset_index()
            plt.plot(grouped[x], grouped[y], marker="o", label=str(label))
        plt.legend(title=hue)
    else:
        grouped = df.groupby(x)[y].mean().reset_index()
        plt.plot(grouped[x], grouped[y], marker="o")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_scatter(df: pd.DataFrame, x: str, y: str, path: Path, title: str):
    sample = df[[x, y, "segment"]].dropna()
    sample.to_csv(path.with_suffix(".csv"), index=False)
    if plt is None:
        _write_svg_scatter(sample, x, y, path, title)
        return

    if len(sample) > 50000:
        sample = sample.sample(50000, random_state=0)
    plt.figure(figsize=(8, 6))
    for segment, group in sample.groupby("segment"):
        plt.scatter(group[x], group[y], s=4, alpha=0.25, label=str(segment))
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.legend(markerscale=3, fontsize="small")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot OPD teacher uncertainty diagnostics.")
    parser.add_argument("diagnostics_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or (args.diagnostics_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    token_df, trajectory_df, high_entropy_df = build_tables(args.diagnostics_dir, max_files=args.max_files)
    if token_df.empty:
        raise SystemExit(f"No trajectory token records found in {args.diagnostics_dir}")

    token_df.to_csv(out_dir / "token_stats.csv", index=False)
    trajectory_df.to_csv(out_dir / "trajectory_stats.csv", index=False)
    high_entropy_df.to_csv(out_dir / "top_teacher_entropy_tokens.csv", index=False)

    save_bar(
        token_df,
        x="segment",
        y="teacher_entropy",
        path=out_dir / "teacher_entropy_by_segment.png",
        title="Teacher Entropy by Segment",
    )
    save_bar(
        token_df,
        x="segment",
        y="logp_gap_teacher_minus_student",
        path=out_dir / "logp_gap_by_segment.png",
        title="Teacher - Student Log Probability Gap by Segment",
    )
    save_line(
        token_df,
        x="turn_id",
        y="teacher_entropy",
        hue="answer_correct",
        path=out_dir / "teacher_entropy_by_turn_correctness.png",
        title="Teacher Entropy by Search Turn",
    )
    save_line(
        token_df,
        x="global_step",
        y="teacher_entropy",
        hue="answer_correct",
        path=out_dir / "teacher_entropy_over_training.png",
        title="Teacher Entropy over Training",
    )
    save_scatter(
        token_df,
        x="teacher_entropy",
        y="logp_gap_teacher_minus_student",
        path=out_dir / "teacher_entropy_vs_logp_gap.png",
        title="Teacher Entropy vs Teacher-Student LogP Gap",
    )

    if token_df["retrieval_hit"].notna().any():
        save_line(
            token_df,
            x="turn_id",
            y="teacher_entropy",
            hue="retrieval_hit",
            path=out_dir / "teacher_entropy_by_turn_retrieval_hit.png",
            title="Teacher Entropy by Turn and Retrieval Hit",
        )

    if plt is None:
        print(f"Wrote diagnostics tables and SVG plots to {out_dir}. Install matplotlib to also render PNG plots.")
    else:
        print(f"Wrote diagnostics tables and plots to {out_dir}")


if __name__ == "__main__":
    main()
