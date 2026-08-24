# Measured values introduced in the preprint that are NOT in the thesis

Every number in the preprint must be attributable. Most are copied verbatim from
`docs/THESIS_FINAL_v2.0_EN.md`. The values below are the exception: they were
measured directly from the project's own artifacts during preprint preparation,
because the thesis reported a quantity that did not match what the experiments
actually consumed. Each row records the command that produced it.

## Dataset sizes (Section 3, `tab:datasets`)

| Value | Meaning | How it was measured |
|---|---|---|
| 32,632 | PathVQA obtained total | `data/pathvqa/*/dataset_info.json` → splits sum (train 19,654 + validation 6,259 + test 6,719) |
| 6,719 | PathVQA evaluated | `results/phase1_baseline/*_pathvqa_seed42.json` → `metadata.num_samples`, all 4 models |
| 7,033 | SLAKE obtained total | `data/slake/*/dataset_info.json` → splits sum (train 4,919 + validation 1,053 + test 1,061) |
| 1,061 | SLAKE evaluated | `results/phase1_baseline/*_slake_seed42.json` → `metadata.num_samples`, all 4 models |
| 2,244 | VQA-RAD obtained total | `data/vqa_rad/*/dataset_info.json` → splits sum (train 1,793 + test 451) |
| 451 | VQA-RAD evaluated | `results/phase1_baseline/*_vqa_rad_seed42.json` → `metadata.num_samples`, all 4 models |

All four models report identical `num_samples` per dataset, and every
`metadata.split` is `test`. No subsampling parameter appears in the metadata, so
the evaluated counts are the full test splits.

## Source-distribution identifiers (Section 3, `tab:datasets`)

Taken verbatim from `src/data/download.py` lines 15, 19, 23:

- `flaviagiammarino/path-vqa`
- `mdwiratathya/SLAKE-vqa-english`
- `flaviagiammarino/vqa-rad`

The SLAKE entry in that file carries the in-repo description
`"SLAKE - English Medical VQA (642 images, ~7K QA pairs)"`, which independently
confirms that the English-only release was the one downloaded.

## Contamination-robustness pair (Section 4.1, `sec:res-phase1`)

The thesis reports the contamination check for Qwen3-VL-2B as
`0.3849 -> 0.3041`. Those two values are stored in
`results/phase1_baseline/phase1_robustness.json`, but they do not reproduce from
the per-sample records that file is derived from. Recomputing with the logic of
the project's own `scripts/robustness_phase1.py` — build `{index: correct}` from
each `results/phase1_baseline/<model>_<dataset>_seed42.json`, remove the union of
Min-K% suspected ids across all four models, and pool sample-weighted — gives:

| Model | full (recomputed) | clean (recomputed) | full (stored) | clean (stored) |
|---|---|---|---|---|
| Qwen3-VL-2B | **0.3843** | **0.3037** | 0.3849 | 0.3041 |
| Qwen2.5-VL-3B | 0.3637 | 0.2765 | 0.3638 | 0.2765 |
| SmolVLM2-2.2B | 0.3391 | 0.2660 | 0.3389 | 0.2662 |
| Gemma4-E2B | 0.1708 | 0.1076 | 0.1721 | 0.1091 |

The removal counts match the stored file exactly (PathVQA 1,020 / SLAKE 233 /
VQA-RAD 73), so the discrepancy is not in the contamination step. It is confined
to PathVQA: the stored file records `acc = 0.348` where both the current
`phase1_baseline` records and the `phase1_baseline_pre_bertscore` copy give
`0.3472`. SLAKE and VQA-RAD reproduce exactly. The stored PathVQA figure
therefore appears to predate a re-scoring of the evaluation records, and is
stale relative to the per-sample data now in the repository.

**Disposition**: the preprint uses the recomputed pair `0.3843 -> 0.3037`. This
also removes the thesis's internal inconsistency, in which the pooled accuracy
of the same model appeared as 0.3843 in Table 4.1a and 0.3849 in the
contamination sentence — with current data both are 0.3843. Every conclusion is
unchanged: the model ranking is preserved position for position before and after
removal, and no delta exceeds 0.0015.

## Discrepancies against the thesis, and how the preprint handles them

| Dataset | Thesis Table 3.2 | Measured | Disposition |
|---|---|---|---|
| PathVQA | 32,799 QA pairs | 32,632 obtained | Both shown; the thesis figure is the published count, kept as "Published" |
| SLAKE | 14,028 QA pairs, "English + Chinese" | 7,033 obtained, English-only | **Substantive**: the language claim does not hold for the copy used. The preprint states the English-only scope explicitly |
| VQA-RAD | 2,248 QA pairs | 2,244 obtained | Both shown; 4-item delta, publication-vs-distribution |

The thesis text also cites "roughly 3,500 question-answer pairs" for VQA-RAD in
its related-work discussion, which is the figure from the original VQA-RAD paper
rather than either of the two counts above. The preprint does not repeat that
figure in the dataset table, to avoid presenting three different numbers for one
quantity.
