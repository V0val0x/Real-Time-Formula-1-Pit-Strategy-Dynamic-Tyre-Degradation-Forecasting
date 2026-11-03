# AI Interaction Record (Phase I)
> **Team Name:** [Your Team Name]
> **Pair:** [Names of Students in AI-Aided Pair]
> **Phase:** I (Black-Box Behavior Investigation)
> **Tool used:** Claude (claude.ai, Opus 5.5)

> Draft started by the AI from its own session. Every row marked *TBD* needs your
> decision and rationale; add a row for each further prompt you send.

---

## 1. Prompt Log & Analytical Decisions

| # | Date | Prompt Summary | AI Suggestion | Decision (Accepted / Modified / Rejected) | Rationale |
|---|------|----------------|---------------|--------------------------------------------|-----------|
| 1 | 2026-11-03 | Gave the repo URL and asked for an explanation of the project. | Summary of the pipeline plus four caveats: metrics are in-sample (no hold-out), `IsVerstappen=0` means "any non-VER driver" not PER, hand-set constants (SC wear factors, pit losses, fuel), small/odd race sample. | *TBD* | *TBD (did you verify each caveat yourselves?)* |
| 2 | 2026-11-03 | "We are the AI-aided pair, help us do this" (the four research questions). | Built `interactive_dashboard.py` + `dashboard_template.html`: 4 tabs, one per question, with filters, sliders and what-if probes (flip driver flag, switch SC/VSC flag on, sweep temperature, sweep tyre age with the engine's own cliff detector). | *TBD* | *TBD* |
| 3 | 2026-11-03 | (same session) | Proposed "what-if" re-prediction (change one input, keep the rest) as the main black-box technique instead of only residual plots. | *TBD* | *TBD: does it stay within Phase I's "inputs/outputs only" rule? We think yes, since no internals are read.* |
| 4 | 2026-11-03 | (same session) | Flagged that every SC, VSC, in-lap and out-lap row has `DegradationDelta = 0` in the training data (baseline is NaN for those rows and gets filled with 0). | *TBD* | *TBD: confirm in the CSV before using in the video.* |
| 5 | | | | | |

---

## 2. Observed AI Failure Modes & Hallucinations

1. **Instance 1: wrong model class**
   * *Context:* First explanation of the repo.
   * *AI Error:* Stated the model is a LightGBM regressor, copying the code's preferred path. The saved artifact is actually scikit-learn's `GradientBoostingRegressor` (the fallback branch, presumably because LightGBM was not installed when it was trained). Found later when the AI loaded the `.joblib` file.
   * *Resolution:* Corrected; dashboard header now prints the real class read from the artifact. Lesson: the AI trusted the source code over the actual artifact.
2. **Instance 2: overlapping bars**
   * *Context:* "Measured label on pit laps" chart with two y-axes.
   * *AI Error:* First version drew the two bar series on top of each other, hiding one.
   * *Resolution:* Caught by the AI on a rendered screenshot; fixed with Plotly `offsetgroup`.
3. **Instance 3: hidden series**
   * *Context:* "Laps until the cliff" chart.
   * *AI Error:* Medium and hard give identical cliff timings, so the medium line was drawn exactly under the hard line and looked missing, which could be misread as "no medium cliff".
   * *Resolution:* Different line widths, dashes and markers per compound.
4. **Instance 4:** *[add your own: anything you found wrong, misleading or over-claimed]*

---

## 3. Reflections & Emerging Best Practices

* **Strengths:** *TBD (e.g. time from question to working interactive dashboard; the AI read the data-labelling code and spotted the zero-label issue quickly)*
* **Limitations:** *TBD (e.g. the model-class error; captions and interpretations the AI wrote must be checked against the plots)*
* **Impact on Confidence:** *TBD*
