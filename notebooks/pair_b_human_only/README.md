# Pair B workspace (Phase I)

**Pair B:** Artur Garipov, Maxim Kalpin

In Phase I we are the human-only pair and used an AI only for the code-completion. In Phase II the roles switch and we work AI-aided.

## Run

```bash
python notebooks/pair_b_human_only/phase1_black_box_analysis.py
```

Saves `visualizations/phase1_pair_b_overview.png` and `visualizations/phase1_pair_b_follow_up.png` and prints the numbers below.

## Findings

Model used as a black box on 963 VER/PER laps (917 on dry tyres), error split by driver, race, tyre and race condition.

1. **Driver gap = one race.** Dry-lap MAE is 0.66 s for PER vs 0.47 s for VER, but almost all of it is PER at Monaco 2023 (~1.9 s). Without Monaco: 0.50 vs 0.46 s. 2024 Bahrain is weak for both drivers.
2. **Wet/intermediate tyres break the model.** 5.7 s MAE on 46 laps vs 0.56 s on dry. The model only knows Soft, Medium and Hard.
3. **Smoother than reality.** On Mediums the prediction follows the trend but misses spikes, and overestimates the first lap on a new set (3.6 s vs 2.8 s).
4. **In-laps have the highest error (1.36 s), but the label is the problem.** The label is 0 on every in-lap, out-lap, SC and VSC lap (no baseline there). Against the driver's median clean lap these laps really cost 13 to 43 s. With no in-lap input the model predicts normal wear (~1.4 s), hence the "error". On out-lap/SC/VSC laps it has flags and learned to predict ~0, so the low errors there just mean it copies the broken label.

**Limits:** the driver input is "VER or not", so "PER" means "not VER". The label also includes traffic. The SC (17) and VSC (11) samples are small. Eight race weekends only.
