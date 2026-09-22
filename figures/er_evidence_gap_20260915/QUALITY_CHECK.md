# Final figure quality checks

PASS after fresh reviewer feedback and primary-agent revisions.

- Values are read from the frozen audited JSON files; source hashes and current plot-script hash match the manifest.
- All six qualifying core cases remain visible; three historical-check failures are open markers, with no pooled mean/CI.
- Both panel (a) axes now spell out correct-minus-wrong margins. The caption specifies the first divergent token.
- The legend has an opaque white background; E002 moved away from the region inequality; text annotations remain readable over reference lines.
- Panel (b) has no in-axis title. Below-axis notes explicitly name the post-hoc common-name comparison and display the negative preregistered full-name visible/ER margins.
- Caption states 40-case stratified review, seven sufficient-evidence errors, and one malformed-label exclusion before fixing six inputs.
- Caption refers to direct evidence visibility and the ER target; no decoding, student-training, prevalence, or causal-effect claim is made.
- Reopened the rendered final PDF via pdftoppm; no clipped labels or text collisions. PNG and SVG also regenerated. Vector PDF has embedded TrueType fonts.
- The integrated introduction matches v3. Sections after the introduction match the prior committed manuscript unchanged.
- Plotting ran in an isolated uv environment; no new model inference or GPU use, and no training configuration edits.

The fresh review remains in FIGURE_REVIEW.md as the review of the earlier version; this file records the disposition of each actionable finding. Pilot status remains unchanged after visual fixes.
