"""
Phase I: Human-Only Pair Starter Analysis.

Guidelines for Pair B:
- Conduct your investigation independently without Generative AI assistance.
- You can use standard documentation and light code completion.
- Focus on understanding model behavior (inputs, outputs, residuals, edge cases).
- Use traditional data science and visualization libraries (Matplotlib, Seaborn, etc.).
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_shared_assets(
    data_path: str = "data/processed/f1_curated_telemetry.csv",
    model_path: str = "src/models/trained_tyre_model.joblib",
):
    """Loads the shared curated dataset and pre-trained degradation model."""
    full_data_path = os.path.join(ROOT_DIR, data_path)
    full_model_path = os.path.join(ROOT_DIR, model_path)

    if not os.path.exists(full_data_path):
        raise FileNotFoundError(f"Dataset not found at {full_data_path}")
    if not os.path.exists(full_model_path):
        raise FileNotFoundError(f"Model not found at {full_model_path}")

    df = pd.read_csv(full_data_path)
    model = joblib.load(full_model_path)
    return df, model


def main():
    print("Loading shared assets...")
    df, model = load_shared_assets()
    print(f"Dataset loaded: {len(df):,} total laps across {df['Race'].nunique()} races.")
    print(f"Drivers: {df['Driver'].unique()}")

    # -------------------------------------------------------------
    # TODO (Pair B - Human-Only):
    # 1. Inspect input features and target degradation metrics.
    # 2. Run model predictions on selected subsets (e.g. Red Bull VER vs PER).
    # 3. Compute residuals: (Actual Degradation - Predicted Degradation).
    # 4. Design and plot your own diagnostic visualizations.
    # 5. Document systematic error patterns, clusters, and edge cases.
    # -------------------------------------------------------------


if __name__ == "__main__":
    main()
