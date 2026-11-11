# Midpoint reflection on Generative AI use (Phase I)

**AI-only pair:**
- Daniel Vuksan, student ID: k________
- Kirill Zaitsev, student ID: k________

**Team:** Claude Nine
**Tool:** Claude (claude.ai)
**Full interaction record:** [`ai-interactions/phase1_ai_log.md`](ai-interactions/phase1_ai_log.md)

## How we used AI

We gave Claude the repository and asked it to explain the project, then to help us answer the four research questions the team agreed on. It built most of the interactive dashboard (`interactive_dashboard.py` and `dashboard_template.html`): four tabs, one per question, with filters, sliders and what-if probes. The what-if idea came from the AI. We hadn't planned it. Re-predicting a lap with one input switched (driver flag, SC flag, VSC flag, temperature) turned out to be the most useful black-box tool we had, because it separates what the model does from the noise in the telemetry.

## Where it helped

- **Speed.** A self-contained Plotly dashboard with linked controls would have taken us much longer to write by hand. The JavaScript and HTML boilerplate alone would have eaten most of our time.
- **Finding data problems.** The AI noticed that every Safety Car, VSC, in-lap and out-lap row has `DegradationDelta = 0` in the training data. We checked this in `f1_data.py` and in the CSV, and it held. It explains why the model predicts almost no degradation on those laps even though they are 15 to 30 s slower.
- **Caveats up front.** In its first explanation it pointed out that the reported metrics are in-sample and that the driver flag means "Verstappen or not", not "Verstappen vs Perez". Both turned out to be true.

## Where it went wrong

- **It trusted the code over the artifact.** It told us the model is LightGBM because that is the preferred branch in the training code. The saved file is a scikit-learn `GradientBoostingRegressor`, the fallback branch. We only found out when the model was actually loaded.
- **First-draft charts had visual bugs.** Two bar series were drawn on top of each other, and the Medium cliff line sat exactly under the Hard line and looked missing. Someone reading the chart could have concluded "no cliff on Mediums".
- **Filtering was incomplete.** The static comparison script did not drop lap 1 (standing start) or yellow-flag laps. That produced a fake +8.5 s spike at tyre age 1 and stretched the residual histogram out to +19 s.

The pattern across all of these: the code looked fine, and the mistakes only showed up in the rendered output or in the actual data. The extra checking took real time, and some of the speed we gained went into it.

## Comparison with Pair B

Pair B (Artur Garipov, Maxim Kalpin) worked from the same four questions, dataset and model, but without using AI for the analysis. Their code and findings are in [`notebooks/pair_b_human_only/`](notebooks/pair_b_human_only/).

**The same key finding, reached from opposite ends.** We found the zero labels by having the AI read the pipeline. Pair B found them from the output side. In-laps had the highest error of any condition, which made no sense for a tyre model, so they checked the label and found it was always zero. Both pairs then compared pit and caution laps with the driver's median green lap in the same race. Those laps are 13 to 43 s slower than normal while the label says 0. Pair B added a twist: the low errors on out-laps, SC and VSC laps are not good news, because the model has flags for them and learned to reproduce the zero.

**What each pair found that the other missed:**

| Pair A (AI-only) | Pair B |
|---|---|
| What-if probes: re-predicting every lap with only the driver, SC or VSC flag switched | Wet and intermediate tyres as a failure case (5.7 s error vs 0.56 s on dry). We filtered to dry compounds at the start and never saw them. |
| Cliff timing per compound from the strategy engine's cliff detector, plus a temperature sweep | The VER/PER error gap is almost entirely Perez at Monaco 2023. Without that race it drops from 0.20 s to 0.05 s. |
| Lap 1 and yellow-flag laps create fake spikes | 2024 Bahrain as a second weak race for both drivers |
| The saved model is a `GradientBoostingRegressor`, not LightGBM | |

**Visualisation choices.** We built one interactive dashboard with four tabs. Pair B made two static Matplotlib figures. The dashboard covers more ground and suits the what-if probes. Pair B's figures are narrower, but every number traces back to one short script.

**Breadth, depth, speed.** AI clearly gave us more breadth: all four questions with several views each, plus interactive probes. Depth was about even. Both pairs went deep on the zero labels, Pair B went deeper on the driver gap, and we went deeper on single-input changes. The dashboard was fast to build, but four AI mistakes had to be found and fixed, and three of them were only visible in the rendered charts.

**Confidence.** Ours comes from being able to probe the model directly. Pair B's comes from sample sizes on every chart and from checking suspicious numbers against the raw CSV. We trust most the findings that both routes produced.

## Direction for Phase II

The roles switch for Phase II. We propose to follow the Phase I findings:

1. **The zero-label effect.** How much do the SC, VSC and out-lap flags matter inside the model, and does that change if the zero-label rows are removed?
2. **Race-specific errors.** Is track temperature a stand-in for "which race is this"? Our temperature sweep and Pair B's Monaco and 2024 Bahrain results both point that way.
3. **The driver flag.** Is it a real effect, or a proxy for something else in the data?

Each pair will use at least two explanation methods and compare them, for example permutation importance, partial dependence, local explanations and nearest-neighbour comparison of laps.

## What we would tell another AI-only pair

- Check claims against the saved artifact and the data, not against the code the AI read.
- Look at every rendered chart yourself. Plausible-looking code still draws wrong pictures.
- Ask the AI for alternatives to the obvious plot. The what-if probes were its best contribution.
- Log the corrections and rejected suggestions, not only the prompts that worked.
