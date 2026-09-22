# Evidence update gap: diagnostic figure

This directory contains a data-driven **pilot / case-study figure**, not a confirmed large-sample mechanism or a student-training result. All measurements come from the completed, audited CPU run `reports/er_correction_cpu_20260914/`. No new model inference was run for this figure.

## Deliverables

- `evidence_update_gap.pdf`: vector, 7.2 × 3.55 inches, embedded TrueType fonts; full-width paper figure.
- `evidence_update_gap.svg`: editable vector with text retained.
- `evidence_update_gap.png`: 300-dpi preview.
- `latex_include.tex`: complete English caption and LaTeX include.
- `plot_evidence_gap.py`: regenerates all figures directly from frozen JSON scores.
- `plotted_cases.csv`: all six plotted states, precision flags, and local margins.
- `candidate_sensitivity.csv`: all four scoring conventions for E017's original and post-hoc names, plus E004.
- `manifest.json`: source hashes, plotting version, and analysis scope.
- `previous_integrated_introduction.md`: preserved prior introduction from the integrated manuscript.
- `FIGURE_REVIEW.md`: fresh figure-review findings; any subsequent fixes recorded in `QUALITY_CHECK.md`.

## Interpretation

Panel (a) asks whether direct evidence access improves the correct token's pairwise odds without making those odds positive. The relevant region is above the diagonal and below the horizontal zero line. E017 is the qualifying example that passes the historical reproduction gate; E004 has the same local pattern only in the separately flagged FP32 sensitivity analysis. E002 is a visible adverse-shift counterexample. The axes compare frozen visibility conditions, not training steps or chronological observations.

Panel (b) uses complete common-name candidates for E017, including their native closing tokens. The short correct name was added after observing the full-reference scores; it is explicitly post hoc. The original long reference does **not** reverse in complete-answer scoring. Four scoring conventions are retained in `candidate_sensitivity.csv` to expose this dependency.

The source experiment's two predefined error-case continuations are still wrong under both visible and ER targets; its correct control remains correct. E017 was not among the predefined continuations. Neither panel establishes open-ended decoding or student-training improvement.

## 中文图注摘要

证据可见性可以改善教师对正确候选的相对支持，却仍不足以翻转错误偏好。(a) 六个固定案例的首个分歧 token 赔率；实心为历史概率复核通过，空心仅为 CPU FP32 内部敏感性。(b) Hemingway 常用姓名的事后完整候选对照：普通教师从隐藏条件的 −3.18 提升到可见条件的 −0.63，ER 进一步到 +1.91，熵匹配仍为 −1.91。原定长名在普通教师和 ER 下均为负，不能将这个别名实例写成稳定纠错或学生收益。

## Reproduction

From the repository root:

```bash
uv run --no-project --with matplotlib==3.11.2 python figures/er_evidence_gap_20260915/plot_evidence_gap.py
```

This uses an isolated uv plotting environment and leaves the training `.venv` unchanged. The script reads source JSONs, checks the completed audit flag, and records their hashes. Review the rendered PDF/PNG, not only the script. No simulated values or extrapolated training curves are used.

## Narrative and next experiment

- `docs/er_opd_introduction_zh_v3.md`: rewritten introduction using current evidence only.
- `docs/paper_sections_1_3_zh.md`: integrated manuscript with the new introduction.
- `docs/er_opd_evidence_gap_experiment_20260915.md`: precise metrics, unselected sampling, controls, falsification criteria, the final three-panel plan, and old-to-new narrative mapping.

The formal follow-on experiments in that document have not been executed. The current pilot should be replaced or supplemented once independent population and generation measurements exist.
