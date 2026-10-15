"""
Red Bull Racing Teammate Telemetry & Model Diagnostic Analysis.
Compares Max Verstappen (VER) vs Sergio Perez (PER) in identical machinery (RB19/RB20):
1. Stint longevity and tyre degradation curves by compound.
2. Model error residuals (VER vs PER) to detect driver-specific biases (Phase I).
3. Strategy window optimization for both drivers (undercut defense vs overcut extension).
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def analyze_red_bull_drivers(
    data_path: str = "data/processed/f1_curated_telemetry.csv",
    model_path: str = "src/models/trained_tyre_model.joblib",
    output_dir: str = "visualizations",
):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    model = joblib.load(model_path)

    # Filter Red Bull drivers on clean dry laps
    rb_df = df[df["Driver"].isin(["VER", "PER"])].copy()
    clean_mask = (~rb_df["IsSafetyCar"]) & (~rb_df["IsVSC"]) & (~rb_df["IsOutLap"]) & (~rb_df["IsInLap"]) & rb_df["LapTimeSeconds"].notna()
    rb_clean = rb_df[clean_mask].copy()

    # Model predictions
    preds = []
    for _, row in rb_clean.iterrows():
        p = model.predict_delta(
            compound=row["Compound"],
            tyre_life=row["TyreLife"],
            effective_tyre_life=row["EffectiveTyreLife"],
            track_temp=row["TrackTemp"],
            fuel_remaining_kg=row["FuelRemainingKg"],
            driver=row["Driver"],
        )
        preds.append(p)

    rb_clean["PredictedDelta"] = preds
    rb_clean["Residual"] = rb_clean["DegradationDelta"] - rb_clean["PredictedDelta"]

    print("=" * 60)
    print("🐂 RED BULL RACING TEAMMATE COMPARISON (VER vs PER) 🐂")
    print("=" * 60)
    for driver in ["VER", "PER"]:
        d_df = rb_clean[rb_clean["Driver"] == driver]
        mae = np.mean(np.abs(d_df["Residual"]))
        rmse = np.sqrt(np.mean(d_df["Residual"] ** 2))
        avg_deg = d_df["DegradationDelta"].mean()
        print(f"Driver {driver}: {len(d_df)} clean laps | Avg Deg Delta: +{avg_deg:.2f}s | Model MAE: {mae:.3f}s | RMSE: {rmse:.3f}s")

    # -------------------------------------------------------------
    # PLOT: VER vs PER Degradation Profile & Residual Discrepancy
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Panel 1: Degradation progression by Tyre Life on MEDIUM
    for driver, color, marker in [("VER", "#0600EF", "o"), ("PER", "#E10600", "s")]:
        d_med = rb_clean[(rb_clean["Driver"] == driver) & (rb_clean["Compound"] == "MEDIUM") & (rb_clean["TyreLife"] <= 30)]
        deg_by_age = d_med.groupby("TyreLife")["DegradationDelta"].mean()
        ax1.plot(deg_by_age.index, deg_by_age.values, label=f"{driver} (Medium Actual)", color=color, marker=marker, linewidth=2, markersize=5)

    ax1.set_xlabel("Tyre Age (Laps)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Mean Degradation Delta (s)", fontsize=12, fontweight="bold")
    ax1.set_title("Red Bull Teammate Degradation Pace (Medium Tyres)", fontsize=13, fontweight="bold")
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Panel 2: Error Residuals Comparison (Model Bias per Driver)
    ver_res = rb_clean[rb_clean["Driver"] == "VER"]["Residual"]
    per_res = rb_clean[rb_clean["Driver"] == "PER"]["Residual"]
    
    ax2.hist(ver_res, bins=25, alpha=0.6, color="#0600EF", label=f"VER (Mean Error: {ver_res.mean():+.2f}s)", density=True)
    ax2.hist(per_res, bins=25, alpha=0.6, color="#E10600", label=f"PER (Mean Error: {per_res.mean():+.2f}s)", density=True)
    ax2.axvline(0, color="black", linestyle="--", linewidth=1)
    
    ax2.set_xlabel("Model Residual: Actual - Predicted (s)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Density", fontsize=12, fontweight="bold")
    ax2.set_title("Model Error Distribution: VER vs PER (Vis4AI Phase I)", fontsize=13, fontweight="bold")
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "phase1_red_bull_comparison.png")
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"\n✓ Saved Red Bull comparative visualization to: {chart_path}")
    print("=" * 60)
    return chart_path


if __name__ == "__main__":
    analyze_red_bull_drivers()
