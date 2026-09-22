# Independent figure quality review

**Verdict:** **Acceptable as an explicitly labeled diagnostic pilot after minor figure edits.** The plotted values and captioned numbers agree with the audited files, and the artifact is careful about the limits of the evidence. It supports the existence of a local mechanism instance: on E017, direct evidence visibility moves the correct-minus-wrong first-token margin from negative toward zero, and the ER target extrapolates the selected short-name pair above zero. It does **not** establish prevalence, decoding improvement, student-training benefit, or a general ER advantage. The current caption states those boundaries, so the missing large-sample and student-training experiments are limitations of the paper's evidence base rather than a reason to reject this pilot figure.

## Required presentation fixes before submission

1. **Remove two text collisions in panel (a).** In the rendered PDF, the vertical zero line passes through the transparent legend, and the `E002` label collides with the region inequality. Give the legend an opaque white background or move it wholly away from the zero line; move `E002` to the right/below its point or move the region annotation farther left/up. These are visible at the intended full-width size.
2. **Put the sign convention on both axes in panel (a).** Change the labels to, for example, `Evidence-hidden first-token margin (correct - wrong; nat)` and `Evidence-visible first-token margin (correct - wrong; nat)`. The caption currently defines the comparison, but `token log odds` alone makes the positive direction less immediate and is less parallel with panel (b).
3. **Remove the title-like text inside panel (b).** `E017: common-name candidate (post-hoc alias sensitivity)` duplicates the caption and functions as an in-panel title. Keep the post-hoc designation in the caption and, if an identifier is needed in the plot, use a compact `E017` annotation beside the bars. This also creates space if the preregistered comparator is added.

## Fairness and selection disclosure

- The filled/open encoding is correct: E002, E008, and E017 pass the historical reproduction check; E003, E004, and E030 are visibly open and labeled as FP32 sensitivity only. No mean or confidence interval mixes the two groups.
- Panel (a) contains all six fixed, analyzable evidence-sufficient errors, and every point matches `plotted_cases.csv`. The phrase `technically eligible` is vague, however. For auditability, say that one of seven evidence-sufficient errors was excluded because of a malformed retrieval label before the six-case core panel was fixed.
- Panel (b) is numerically correct and explicitly marked post hoc. It shows the favorable short-name alias while the preregistered long-name result is relegated to the caption. This is truthful but visually selective. The strongest submission version would show two candidate rows/groups—`preregistered full name` and `post-hoc common name`—with the four methods in each, or at minimum add a compact gray callout giving the preregistered visible/ER margins (`-30.21`, `-27.37`). If space prevents this, the existing explicit caption disclosure is the minimum acceptable treatment; do not describe the panel as a primary correction result.
- E004 is a useful corroborating local example but fails historical parity. The caption's open-marker rule makes this clear. If naming E004 in prose, append `FP32 sensitivity only` so it is not read as equal-strength evidence to E017.

## Caption and claim audit

The caption is substantially self-contained. It states the unit and size of the panel (six student-error cases), the first-divergence metric, the shared-prefix comparison, the historical-check split (`n=3` and `n=3` in the legend), the post-hoc alias selection, the preregistered negative result, the frozen teacher and histories, the masking limitation, and the claims the figure cannot support.

All checked numerical and directional statements are correct:

- E017 and E004 satisfy `m_hid < m_obs < 0`; only E017 is a historical-check pass.
- E002 shifts adversely (`-2.42` to `-8.70`).
- For the E017 post-hoc common-name full sequence, hidden/visible/ER/entropy-matched margins round to `-3.18/-0.63/+1.91/-1.91`.
- For the preregistered `Ernest Miller Hemingway` sequence, visible and ER full-answer margins round to `-30.21` and `-27.37` and remain negative.

Two wording changes would make the causal and algorithmic scope more precise:

- Replace the bold lead with **`Making retrieved evidence directly visible can shift a frozen teacher toward the correct candidate without reversing its preference.`** This matches the actual visibility intervention and avoids reading `evidence` as a pure causal contribution; the student history may already carry evidence.
- In panel (b), write **`the ER target (alpha=1) reverses this selected pairwise ranking`** rather than `ER reverses it`. This makes clear that the bar is a constructed teacher target score, not a trained student's output or a decoding result.

The separate fixed-generation check did not rescue either of the two erroneous continuations, and no student was trained. The caption already says the panels do not show decoding accuracy or student-training gains. That boundary is adequate; the generation detail can remain in the main text rather than lengthening the caption further.

## Render and reproducibility checks

- The PDF and PNG contain the same visible content. There is no edge clipping, the value labels are inside the page, and the 8--9 pt type is readable at the declared 7.2-inch full-width size.
- The PDF is vector output with embedded DejaVu Serif/STIX fonts.
- Fill state plus numeric labels preserve meaning without color, and the bar categories do not rely on color alone.
- `manifest.json` hashes match the current plotting script, completion marker, audit, score JSONL, and alias JSON. The audited recomputation error is zero; maximum entropy gaps are below `1.7e-6` nat; CUDA was not initialized.

## Recommended disposition

Use the figure in a submission only as a **pilot/mechanism diagnostic** and retain the negative canonical-name and no-training language. After fixing the two collisions and axis terminology, it is visually submission-ready. Adding the preregistered comparator to panel (b) would materially improve resistance to a selection-bias objection, but the current explicit post-hoc label and negative-result disclosure keep the figure truthful if layout constraints rule that out.
