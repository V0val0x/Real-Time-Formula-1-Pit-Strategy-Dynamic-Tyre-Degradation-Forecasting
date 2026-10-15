"""
Model Training Script for F1 Real Telemetry Dataset.
Trains TyreDegradationModel on curated real race data (2023-2024),
evaluates metrics across compounds, and saves the trained model artifact.
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import joblib
import pandas as pd
from src.models.tyre_model import TyreDegradationModel


def train_and_save_model(
    data_path: str = "data/processed/f1_curated_telemetry.csv",
    model_save_path: str = "src/models/trained_tyre_model.joblib",
):
    print("=" * 65)
    print("🏎️ TRAINING F1 TYRE DEGRADATION MODEL ON REAL 2023-2024 RACES 🏎️")
    print("=" * 65)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run download_real_data.py first.")

    df = pd.read_csv(data_path)
    print(f"\n[1/3] Loaded dataset: {len(df):,} total laps across {df['Race'].nunique()} Grand Prix.")
    print(f"      Compounds present: {df['Compound'].value_counts().to_dict()}")

    # Filter out invalid / pit lap outliers for clean degradation training
    train_mask = (df["LapTimeSeconds"].notna()) & (df["DegradationDelta"].notna()) & (df["Compound"].isin(["SOFT", "MEDIUM", "HARD"]))
    clean_df = df[train_mask].copy()
    print(f"      Clean training samples: {len(clean_df):,} laps.")

    # Train model
    print("\n[2/3] Fitting Physics-Aware Degradation Model...")
    model = TyreDegradationModel()
    metrics = model.fit(clean_df)
    
    print("\n[3/3] Training Evaluation:")
    print(f"      • RMSE : {metrics['rmse']:.3f} s")
    print(f"      • MAE  : {metrics['mae']:.3f} s")
    print(f"      • R²   : {metrics['r2']:.3f}")

    print("\n📊 Feature Importances (What did the model prioritize?):")
    for feat, imp in model.get_feature_importance().items():
        bar = "█" * int(imp * 35)
        print(f"      {feat:<20} : {imp*100:5.1f}% {bar}")

    # Save model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)
    print(f"\n✅ Model saved to: {model_save_path}")
    print("=" * 65)
    return model


if __name__ == "__main__":
    train_and_save_model()
