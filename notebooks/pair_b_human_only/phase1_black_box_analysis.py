"""Phase I, Pair B: black-box analysis of the tyre model (inputs in, predictions out).

Run from repo root: python notebooks/pair_b_human_only/phase1_black_box_analysis.py
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

os.environ.setdefault("MPLCONFIGDIR", os.path.join(ROOT_DIR, ".matplotlib"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "f1_curated_telemetry.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "src", "models", "trained_tyre_model.joblib")
OUTPUT_DIR = os.path.join(ROOT_DIR, "visualizations")
DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
DRIVER_COLORS = {"VER": "#0600EF", "PER": "#E10600"}
SPECIAL_CONDITIONS = ["In lap", "Out lap", "Safety Car", "VSC"]


def condition_name(row):
    """One condition label per lap."""
    if row["IsSafetyCar"]:
        return "Safety Car"
    if row["IsVSC"]:
        return "VSC"
    if row["IsOutLap"]:
        return "Out lap"
    if row["IsInLap"]:
        return "In lap"
    return "Green"


def predict_laps(model, rows):
    predictions = []
    for _, row in rows.iterrows():
        prediction = model.predict_delta(
            compound=row["Compound"],
            tyre_life=row["TyreLife"],
            effective_tyre_life=row["EffectiveTyreLife"],
            track_temp=row["TrackTemp"],
            fuel_remaining_kg=row["FuelRemainingKg"],
            driver=row["Driver"],
            is_sc=row["IsSafetyCar"],
            is_vsc=row["IsVSC"],
            is_out_lap=row["IsOutLap"],
        )
        predictions.append(prediction)
    return predictions


def add_time_lost(red_bull):
    # pit/caution laps have no baseline -> compare to driver's median clean lap in that race
    clean = red_bull[
        (red_bull["Condition"] == "Green")
        & red_bull["Compound"].isin(DRY_COMPOUNDS)
        & ~red_bull["IsYellow"]
        & (red_bull["LapNumber"] > 1)
    ]
    reference = clean.groupby(["RaceEvent", "Driver"])["LapTimeSeconds"].median()
    reference = reference.rename("ReferenceLap").reset_index()
    red_bull = red_bull.merge(reference, on=["RaceEvent", "Driver"], how="left")
    red_bull["TimeLost"] = red_bull["LapTimeSeconds"] - red_bull["ReferenceLap"]
    return red_bull


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    # VER vs PER only
    red_bull = data[data["Driver"].isin(["VER", "PER"])].copy()
    red_bull = red_bull[red_bull["LapTimeSeconds"].notna()].copy()
    red_bull["Prediction"] = predict_laps(model, red_bull)
    red_bull["Residual"] = red_bull["DegradationDelta"] - red_bull["Prediction"]
    red_bull["AbsoluteError"] = red_bull["Residual"].abs()
    red_bull["Condition"] = red_bull.apply(condition_name, axis=1)
    red_bull["RaceEvent"] = red_bull["Year"].astype(str) + " " + red_bull["Race"]
    red_bull["CompoundSupport"] = np.where(
        red_bull["Compound"].isin(DRY_COMPOUNDS), "Dry (supported)", "Wet/intermediate"
    )
    red_bull = add_time_lost(red_bull)

    dry = red_bull[red_bull["Compound"].isin(DRY_COMPOUNDS)].copy()
    assert len(red_bull) > 900 and set(red_bull["Driver"]) == {"VER", "PER"}

    # figure 1: overview
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    driver_stats = dry.groupby("Driver").agg(MAE=("AbsoluteError", "mean"), Laps=("Residual", "size"))
    driver_stats["MAE"].plot.bar(
        ax=axes[0, 0], color=[DRIVER_COLORS[d] for d in driver_stats.index], rot=0
    )
    axes[0, 0].set_title("Dry-lap error by driver")
    axes[0, 0].set_ylabel("Mean absolute error (s)")
    for i, (_, row) in enumerate(driver_stats.iterrows()):
        axes[0, 0].text(i, row["MAE"] + 0.02, f"n={int(row['Laps'])}", ha="center")
    axes[0, 0].set_ylim(0, driver_stats["MAE"].max() * 1.15)

    race_driver = dry.groupby(["RaceEvent", "Driver"])["AbsoluteError"].mean().unstack()
    race_driver.plot.bar(ax=axes[0, 1], color=DRIVER_COLORS)
    axes[0, 1].set_title("Dry-lap error by race event")
    axes[0, 1].set_ylabel("Mean absolute error (s)")
    axes[0, 1].set_xlabel("")
    axes[0, 1].tick_params(axis="x", rotation=45)

    # clip to +-4 s, a few outliers hide the shape
    outside = (dry["Residual"].abs() > 4).sum()
    for driver in ["VER", "PER"]:
        axes[1, 0].hist(
            dry.loc[dry["Driver"] == driver, "Residual"],
            bins=40,
            range=(-4, 4),
            alpha=0.55,
            density=True,
            label=driver,
            color=DRIVER_COLORS[driver],
        )
    axes[1, 0].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_title(f"Dry-lap residuals ({outside} laps beyond +-4 s not shown)")
    axes[1, 0].set_xlabel("Label - prediction (s)")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].legend()

    support_stats = red_bull.groupby("CompoundSupport").agg(
        MAE=("AbsoluteError", "mean"), Laps=("Residual", "size")
    )
    support_stats["MAE"].plot.bar(ax=axes[1, 1], color=["#4CAF50", "#FF9800"], rot=0)
    axes[1, 1].set_title("Dry tyres vs wet/intermediate tyres")
    axes[1, 1].set_ylabel("Mean absolute error (s)")
    axes[1, 1].set_xlabel("")
    for i, (_, row) in enumerate(support_stats.iterrows()):
        axes[1, 1].text(i, row["MAE"] + 0.05, f"n={int(row['Laps'])}", ha="center")

    for ax in axes.flat:
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.suptitle("Pair B, Phase I: how the tyre-degradation model behaves (VER vs PER)", fontsize=16)
    fig.tight_layout()
    overview_path = os.path.join(OUTPUT_DIR, "phase1_pair_b_overview.png")
    fig.savefig(overview_path, dpi=180)
    plt.close(fig)

    # figure 2: follow-up
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    medium = dry[(dry["Compound"] == "MEDIUM") & (dry["TyreLife"] <= 30)]
    age_curve = medium.groupby("TyreLife")[["DegradationDelta", "Prediction"]].mean()
    axes[0].plot(age_curve.index, age_curve["DegradationDelta"], marker="o", label="Label")
    axes[0].plot(age_curve.index, age_curve["Prediction"], linestyle="--", label="Prediction")
    axes[0].set_title("Medium tyre: average by tyre age")
    axes[0].set_xlabel("Tyre age (laps)")
    axes[0].set_ylabel("Degradation delta (s)")
    axes[0].legend()

    condition_stats = red_bull.groupby("Condition").agg(
        MAE=("AbsoluteError", "mean"), Laps=("Residual", "size")
    ).sort_values("MAE")
    condition_stats["MAE"].plot.barh(ax=axes[1], color="#607D8B")
    axes[1].set_title("Error by race condition")
    axes[1].set_xlabel("Mean absolute error (s)")
    axes[1].set_ylabel("")
    for i, (_, row) in enumerate(condition_stats.iterrows()):
        axes[1].text(row["MAE"] + 0.02, i, f"n={int(row['Laps'])}", va="center")

    # high in-lap error -> check the label itself
    special = red_bull[red_bull["Condition"].isin(SPECIAL_CONDITIONS)]
    label_check = special.groupby("Condition")[["DegradationDelta", "Prediction", "TimeLost"]].mean()
    label_check = label_check.reindex(SPECIAL_CONDITIONS)
    label_check.columns = ["Label in dataset", "Model prediction", "Real time lost vs clean lap"]
    label_check.plot.bar(ax=axes[2], rot=0, color=["#9E9E9E", "#FF9800", "#3F51B5"])
    axes[2].set_title("Pit and caution laps: label vs prediction vs reality")
    axes[2].set_ylabel("Seconds (mean)")
    axes[2].set_xlabel("")
    axes[2].set_yscale("symlog", linthresh=2)
    for i in range(len(SPECIAL_CONDITIONS)):
        axes[2].text(i - 0.17, 0.15, "label = 0", rotation=90, ha="center", va="bottom", fontsize=9)

    for ax in axes:
        ax.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    detail_path = os.path.join(OUTPUT_DIR, "phase1_pair_b_follow_up.png")
    fig.savefig(detail_path, dpi=180)
    plt.close(fig)

    # numbers for the README
    label_counts = special.groupby("Condition").agg(
        Laps=("DegradationDelta", "size"),
        LabelZeroShare=("DegradationDelta", lambda s: (s == 0).mean()),
        MeanTyreAge=("TyreLife", "mean"),
    )
    print(f"Analysed {len(red_bull)} Red Bull laps ({len(dry)} on dry compounds).\n")
    print("Dry error by driver:\n", dry.groupby("Driver")[["AbsoluteError", "Residual"]].mean().round(3), "\n")
    no_monaco = dry[dry["RaceEvent"] != "2023 Monaco"]
    print("Dry MAE by driver without 2023 Monaco:\n", no_monaco.groupby("Driver")["AbsoluteError"].mean().round(3), "\n")
    print("Dry MAE by race event:\n", dry.groupby("RaceEvent")["AbsoluteError"].mean().sort_values(ascending=False).round(3), "\n")
    print("By compound support:\n", support_stats.round(3), "\n")
    print("By condition:\n", condition_stats.round(3), "\n")
    print("Pit/caution label check:\n", label_check.round(2), "\n", label_counts.round(2), "\n")
    print(f"Saved: {overview_path}\nSaved: {detail_path}")


if __name__ == "__main__":
    main()
