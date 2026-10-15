"""
FastF1 Batch Data Extractor and Processor.
Downloads and pre-processes curated F1 race sessions representing
diverse tyre degradation archetypes (High-Deg, Front-Limited, Low-Deg Street, High-Speed).
"""

from __future__ import annotations
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import argparse
import pandas as pd
from src.data.f1_data import F1DataLoader


# Curated representative Grand Prix sessions
CURATED_RACES = [
    (2023, "Bahrain"),        # Extreme rear tyre thermal degradation, abrasive asphalt
    (2023, "Spain"),          # High lateral load & high tyre wear
    (2023, "Great Britain"),  # High speed front-limited tyre stress (Silverstone)
    (2023, "Monza"),          # Low downforce, minimal braking wear, high straight speed
    (2023, "Monaco"),         # Extreme low degradation street circuit
    (2023, "Austria"),        # High power, short lap, multi-stop strategy dynamics
    (2024, "Bahrain"),        # Cross-year validation: current season baseline
    (2024, "Japan"),          # Suzuka high-energy S-curves
]


def download_and_process_dataset(
    races: list[tuple[int, str]] = None,
    output_path: str = "data/processed/f1_curated_telemetry.csv",
    cache_dir: str = "data/cache",
) -> pd.DataFrame:
    if races is None:
        races = CURATED_RACES

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    loader = F1DataLoader(cache_dir=cache_dir)

    all_dfs = []
    print(f"🚀 Starting batch download for {len(races)} representative races...")
    
    for idx, (year, race_name) in enumerate(races, 1):
        print(f"\n[{idx}/{len(races)}] Loading {year} {race_name} Grand Prix...")
        try:
            df = loader.load_race_session(year, race_name)
            print(f"   ✓ Extracted {len(df)} laps across {df['Driver'].nunique()} drivers.")
            all_dfs.append(df)
        except Exception as e:
            print(f"   ⚠️ Warning: Failed to load {year} {race_name}: {e}")

    if not all_dfs:
        raise RuntimeError("No race data could be loaded.")

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.to_csv(output_path, index=False)
    print("\n" + "=" * 60)
    print(f"✅ SUCCESS: Combined dataset saved to: {output_path}")
    print(f"Total laps: {len(combined_df):,} | Circuits: {combined_df['Race'].nunique()}")
    print("=" * 60)
    return combined_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download real FastF1 race data.")
    parser.add_argument("--all", action="store_true", help="Download all races instead of curated set")
    args = parser.parse_args()

    download_and_process_dataset()
