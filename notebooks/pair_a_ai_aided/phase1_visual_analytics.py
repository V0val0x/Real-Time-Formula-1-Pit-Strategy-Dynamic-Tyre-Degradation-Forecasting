"""
Visual Analytics Generator for Phase I (How does the model behave?).
Generates multi-panel diagnostic visualizations based on REAL 2023 Bahrain GP Telemetry:
1. Actual vs Predicted Degradation Profiles across Stints (VER).
2. Red Bull Teammate Head-to-Head: VER vs PER Stint Lengths & Tyre Cliffs.
3. Real-Time Strategy Decision & Reactive Contingency Timeline.
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.models.tyre_model import TyreDegradationModel
from src.simulation.race_simulator import RaceSimulator


def generate_phase1_visualizations(
    data_path: str = "data/processed/f1_curated_telemetry.csv",
    model_path: str = "src/models/trained_tyre_model.joblib",
    output_dir: str = "visualizations",
):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Real Telemetry
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing {data_path}. Run scripts/download_real_data.py first.")

    all_df = pd.read_csv(data_path)
    race_df = all_df[(all_df["Year"] == 2023) & (all_df["Race"] == "Bahrain")].copy()
    
    # 2. Load Pre-Trained Model
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        model = TyreDegradationModel()
        model.fit(all_df)

    # 3. Predict for Max Verstappen (VER) on Real Laps
    ver_df = race_df[race_df["Driver"] == "VER"].sort_values("LapNumber").copy()
    preds = []
    for _, row in ver_df.iterrows():
        p = model.predict_delta(
            compound=row["Compound"],
            tyre_life=row["TyreLife"],
            effective_tyre_life=row["EffectiveTyreLife"],
            track_temp=row["TrackTemp"],
            fuel_remaining_kg=row["FuelRemainingKg"],
            driver="VER",
            is_sc=row["IsSafetyCar"],
            is_vsc=row["IsVSC"],
            is_out_lap=row["IsOutLap"]
        )
        preds.append(p)

    ver_df["PredictedDelta"] = preds
    ver_df["Residual"] = ver_df["DegradationDelta"] - ver_df["PredictedDelta"]

    # -------------------------------------------------------------
    # PLOT 1: Real Lap-by-Lap Degradation & Residual Diagnostic (VER)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # Panel 1: Real Telemetry vs Model Prediction
    ax1.plot(ver_df["LapNumber"], ver_df["DegradationDelta"], "o-", color="#E10600", label="Real Bahrain Telemetry (Degradation Delta)", linewidth=2, markersize=4)
    ax1.plot(ver_df["LapNumber"], ver_df["PredictedDelta"], "--", color="#1E88E5", label="Trained ML Model Prediction", linewidth=2.5)

    # Annotate Stints (Soft -> Soft -> Hard)
    ax1.axvline(14.5, color="black", linestyle="--", alpha=0.7, label="Pit Stop 1 (Lap 14: Soft -> Soft)")
    ax1.axvline(36.5, color="black", linestyle=":", alpha=0.7, label="Pit Stop 2 (Lap 36: Soft -> Hard)")

    # Real VSC window (Leclerc retirement on lap 41-42)
    ax1.axvspan(40.5, 42.5, color="gold", alpha=0.35, label="Real VSC Window (Laps 41-42: LEC DNF)")

    ax1.set_ylabel("Degradation Delta (seconds)", fontsize=12, fontweight="bold")
    ax1.set_title("Vis4AI Phase I: Real Bahrain GP Telemetry vs Model Prediction (VER - Red Bull RB19)", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Panel 2: Error Residuals
    ax2.bar(ver_df["LapNumber"], ver_df["Residual"], color=np.where(ver_df["Residual"] > 0, "#D32F2F", "#388E3C"), alpha=0.75)
    ax2.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax2.set_xlabel("Race Lap Number", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Residual (s)", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    chart1_path = os.path.join(output_dir, "phase1_degradation_behavior.png")
    plt.savefig(chart1_path, dpi=200)
    plt.close()
    print(f"✓ Saved REAL Telemetry diagnostic plot to: {chart1_path}")

    # -------------------------------------------------------------
    # PLOT 2: Real Teammate Head-to-Head (VER vs PER) Stint Analysis
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(15, 6))
    per_df = race_df[race_df["Driver"] == "PER"].sort_values("LapNumber").copy()

    ax.plot(ver_df["LapNumber"], ver_df["DegradationDelta"], "o-", color="#0600EF", label="VER (Stints: S14 / S36 / H57)", linewidth=2, markersize=4)
    ax.plot(per_df["LapNumber"], per_df["DegradationDelta"], "s-", color="#E10600", label="PER (Stints: S17 / S34 / H57)", linewidth=2, markersize=4, alpha=0.85)

    ax.axvspan(40.5, 42.5, color="gold", alpha=0.3, label="VSC Period (Laps 41-42)")
    ax.set_xlabel("Race Lap Number", fontsize=12, fontweight="bold")
    ax.set_ylabel("Degradation Delta (s)", fontsize=12, fontweight="bold")
    ax.set_title("Red Bull Teammate Real Strategy Divergence: VER vs PER (2023 Bahrain GP)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    chart2_path = os.path.join(output_dir, "phase1_red_bull_head_to_head.png")
    plt.savefig(chart2_path, dpi=200)
    plt.close()
    print(f"✓ Saved Red Bull Head-to-Head plot to: {chart2_path}")

    # -------------------------------------------------------------
    # PLOT 3: Real-Time Strategy Decisions on Real Race Replay
    # -------------------------------------------------------------
    sim = RaceSimulator(race_df=race_df, target_driver="VER", model=model)
    sim_df = sim.run_replay()

    fig, ax = plt.subplots(figsize=(15, 5))
    plan_colors = {
        "Plan A (Stint Management)": "#4CAF50",
        "Plan A (Nominal Target Reached)": "#2196F3",
        "Plan A (Stint 2 to Finish)": "#009688",
        "Plan VSC (Opportunistic Cheap Stop)": "#FFC107",
        "Plan U (Undercut Defense)": "#9C27B0",
        "Plan C (Cliff Mitigation)": "#F44336",
    }
    
    for _, row in sim_df.iterrows():
        lap = row["LapNumber"]
        plan = row["ActivePlan"]
        color = plan_colors.get(plan, "#9E9E9E")
        ax.bar(lap, 1, width=0.9, color=color, edgecolor="none", alpha=0.85)

    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for col in plan_colors.values()]
    ax.legend(handles, plan_colors.keys(), loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=True)
    ax.set_yticks([])
    ax.set_xlabel("Race Lap", fontsize=12, fontweight="bold")
    ax.set_title("Strategy Decision Adaptation on Real 2023 Bahrain GP (VER)", fontsize=14, fontweight="bold")
    ax.set_xlim(1, 57)

    plt.tight_layout()
    chart3_path = os.path.join(output_dir, "phase1_strategy_timeline.png")
    plt.savefig(chart3_path, dpi=200)
    plt.close()
    print(f"✓ Saved Real Race Strategy Timeline plot to: {chart3_path}")

    return chart1_path, chart2_path, chart3_path


if __name__ == "__main__":
    generate_phase1_visualizations()
