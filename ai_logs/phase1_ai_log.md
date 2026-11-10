# AI Interaction Record (Phase I)
> **Team Name:** Claude Nine
> **Pair:** Daniel Vuksan, Kirill Zaitsev
> **Phase:** I (Black-Box Behavior Investigation)
> **Tool used:** Claude (claude.ai)

---

## 1. Prompt Log & Analytical Decisions

| # | Date | Prompt Summary | AI Suggestion | Decision (Accepted / Modified / Rejected) | Rationale |
|---|------|----------------|---------------|--------------------------------------------|-----------|
| 1 | 2026-11-03 | Gave the repo URL and asked for an explanation of the project. | Summary of the pipeline plus four caveats: metrics are in-sample (no hold-out), `IsVerstappen=0` means "any non-VER driver" not PER, hand-set constants (SC wear factors, pit losses, fuel), small/odd race sample. | Accepted (after manual check) | We checked `train_model.py` and `tyre_model.py` ourselves and confirmed all four points. However, we caught that the AI initially named the wrong model class (see Failure Mode 1). |
| 2 | 2026-11-03 | "We are the AI-aided pair, help us do this" (the four research questions). | Built `interactive_dashboard.py` + `dashboard_template.html`: 4 tabs, one per question, with filters, sliders and what-if probes (flip driver flag, switch SC/VSC flag on, sweep temperature, sweep tyre age with the engine's own cliff detector). | Modified | Accepted the 4-tab layout so it maps directly to our 4 research questions, but we had to fix overlapping bars and hidden lines in the generated Plotly charts. |
| 3 | 2026-11-03 | (same session) | Proposed "what-if" re-prediction (change one input, keep the rest) as the main black-box technique instead of only residual plots. | Accepted | Good fit for Phase I: we only vary input columns and observe output predictions without reading tree weights or internal splits, and it separates model behavior from noisy telemetry. |
| 4 | 2026-11-03 | (same session) | Flagged that every SC, VSC, in-lap and out-lap row has `DegradationDelta = 0` in the training data (baseline is NaN for those rows and gets filled with 0). | Accepted | Verified directly in `f1_curated_telemetry.csv` and `f1_data.py`. This explains why the model predicts near-zero degradation on caution and pit laps even when actual lap times are 15–30s slower. |
| 5 | 2026-11-10 | Asked to review static scripts (`red_bull_analysis.py`) and check for remaining issues before submission. | Pointed out that `clean_mask` in `red_bull_analysis.py` did not filter out `LapNumber == 1` (standing start) or `IsYellow`, causing an artificial +8.5s spike at TyreLife=1 and outliers up to +19s in the histogram. | Accepted | Filtering out lap 1 and yellow flags makes the Medium tyre comparison and residual histogram readable and consistent with the interactive dashboard. |

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
4. **Instance 4: unfiltered Lap 1 and yellow-flag outliers in static plots**
   * *Context:* Static comparison script `red_bull_analysis.py` (`phase1_red_bull_comparison.png`).
   * *AI Error:* The initial `clean_mask` only excluded SC, VSC, in-laps, and out-laps, leaving in Lap 1 (standing start) and yellow-flag laps. This created a fake +8.5s degradation spike at tyre age 1 for PER and stretched the residual histogram x-axis out to +19s.
   * *Resolution:* Added `LapNumber > 1` and `~IsYellow` to the filter and bounded the histogram range so the actual VER vs PER error distribution is clearly visible.

---

## 3. Reflections & Emerging Best Practices

* **Strengths:** AI made it much faster to build a self-contained HTML/Plotly dashboard with sliders and linked filters than writing all the JS/HTML boilerplate by hand. It also helped spot data-pipeline quirks quickly, like `DegradationDelta` being filled with `0.0` on SC/VSC and pit laps.
* **Limitations:** AI often trusts code comments or default branches without checking the actual saved data/artifacts (like assuming LightGBM instead of `GradientBoostingRegressor`), and its first-pass plots frequently had visual bugs (overlapping bars, hidden lines, unfiltered Lap 1 outliers) that only showed up when inspecting the rendered charts.
* **Impact on Confidence:** Having interactive "what-if" probes alongside raw residual plots made us more confident in our conclusions, because we could directly test how the model reacts when a single variable (driver flag, SC flag, or temperature) is changed while holding everything else constant. However, we learned we had to verify every claim against the CSV and rendered plots first.
