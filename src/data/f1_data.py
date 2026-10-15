"""
F1 Data Loader and Feature Engineering Module.
Supports both FastF1 live/cached data extraction and a realistic physics-compliant synthetic generator.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any

try:
    import fastf1
    FASTF1_AVAILABLE = True
except ImportError:
    FASTF1_AVAILABLE = False


class F1DataLoader:
    """Loads and preprocesses Formula 1 race session data."""

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir
        if FASTF1_AVAILABLE:
            os.makedirs(cache_dir, exist_ok=True)
            fastf1.Cache.enable_cache(cache_dir)

    def load_race_session(self, year: int, race: str | int) -> pd.DataFrame:
        """
        Loads a race session via FastF1 and extracts clean lap-by-lap features.
        """
        if not FASTF1_AVAILABLE:
            raise RuntimeError("FastF1 is not installed or available.")

        session = fastf1.get_session(year, race, "R")
        session.load(telemetry=False, weather=True)

        laps = session.laps
        weather = session.weather_data

        records = []
        for driver in laps["Driver"].unique():
            driver_laps = laps.pick_drivers(driver).sort_values("LapNumber")
            
            eff_wear = 0.0
            prev_stint = None
            total_fuel = 105.0  # kg
            est_fuel_per_lap = total_fuel / max(1, len(driver_laps))

            for _, lap in driver_laps.iterrows():
                lap_no = int(lap["LapNumber"])
                compound = str(lap.get("Compound", "UNKNOWN")).upper()
                tyre_life = float(lap.get("TyreLife", 1.0)) if not pd.isna(lap.get("TyreLife")) else 1.0
                stint = int(lap.get("Stint", 1)) if not pd.isna(lap.get("Stint")) else 1
                
                # Pit in/out
                is_out_lap = pd.notna(lap.get("PitOutTime"))
                is_in_lap = pd.notna(lap.get("PitInTime"))

                # Reset effective wear when changing stints or on fresh set
                if (prev_stint is not None and stint != prev_stint) or is_out_lap or tyre_life == 1.0:
                    eff_wear = 0.0
                prev_stint = stint

                # Lap time in seconds
                lap_time = lap.get("LapTime")
                lap_time_s = lap_time.total_seconds() if pd.notna(lap_time) else np.nan

                # Track status interpretation
                track_status = str(lap.get("TrackStatus", "1"))
                is_sc = "4" in track_status
                is_vsc = "6" in track_status
                is_yellow = "2" in track_status
                is_clean = not (is_sc or is_vsc or is_yellow)

                # Effective tyre wear increment
                if is_sc:
                    eff_wear += 0.20
                elif is_vsc:
                    eff_wear += 0.35
                elif is_yellow:
                    eff_wear += 0.60
                else:
                    eff_wear += 1.0

                # Track temperature
                track_temp = 32.0
                air_temp = 25.0
                if weather is not None and not weather.empty:
                    match_weather = weather[weather["Time"] <= lap.get("Time", pd.Timedelta(0))]
                    if not match_weather.empty:
                        track_temp = float(match_weather.iloc[-1].get("TrackTemp", 32.0))
                        air_temp = float(match_weather.iloc[-1].get("AirTemp", 25.0))

                records.append({
                    "Year": year,
                    "Race": str(race),
                    "Driver": driver,
                    "LapNumber": lap_no,
                    "LapTimeSeconds": lap_time_s,
                    "Compound": compound,
                    "TyreLife": tyre_life,
                    "EffectiveTyreLife": round(eff_wear, 2),
                    "Stint": stint,
                    "TrackStatus": track_status,
                    "IsSafetyCar": is_sc,
                    "IsVSC": is_vsc,
                    "IsYellow": is_yellow,
                    "IsOutLap": is_out_lap,
                    "IsInLap": is_in_lap,
                    "TrackTemp": track_temp,
                    "AirTemp": air_temp,
                    "FuelRemainingKg": max(2.0, total_fuel - (lap_no * est_fuel_per_lap)),
                    "Position": int(lap.get("Position", 1)) if pd.notna(lap.get("Position")) else 1,
                })

        df = pd.DataFrame(records)
        df = self._add_relative_metrics(df)
        return df

    def _add_relative_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds cumulative race time, gaps to car ahead/behind and baseline delta."""
        df = df.sort_values(["Year", "Race", "Driver", "LapNumber"]).reset_index(drop=True)

        # 1. Cumulative race time per driver
        df["CumulativeTime"] = df.groupby(["Year", "Race", "Driver"])["LapTimeSeconds"].cumsum()

        # 2. Position by cumulative time per lap
        df = df.sort_values(["Year", "Race", "LapNumber", "Position"]).reset_index(drop=True)

        # 3. Gaps
        leader_time = df.groupby(["Year", "Race", "LapNumber"])["CumulativeTime"].transform("min")
        df["IntervalToLeader"] = (df["CumulativeTime"] - leader_time).round(3).fillna(0.0)

        df["GapAhead"] = (
            df.groupby(["Year", "Race", "LapNumber"])["CumulativeTime"].diff().fillna(0.0).round(3)
        )
        df["GapBehind"] = (
            (-df.groupby(["Year", "Race", "LapNumber"])["CumulativeTime"].diff(-1)).fillna(99.0).round(3)
        )

        # 4. Degradation delta relative to clean driver baseline per compound
        clean_laps = df[(~df["IsSafetyCar"]) & (~df["IsVSC"]) & (~df["IsOutLap"]) & (~df["IsInLap"]) & df["LapTimeSeconds"].notna()]
        if not clean_laps.empty:
            driver_min = clean_laps.groupby(["Year", "Race", "Driver", "Compound"])["LapTimeSeconds"].transform("min")
            df["BaselineLapTime"] = driver_min
            df["DegradationDelta"] = (df["LapTimeSeconds"] - df["BaselineLapTime"]).fillna(0.0).clip(lower=0.0).round(3)
        else:
            df["BaselineLapTime"] = df["LapTimeSeconds"]
            df["DegradationDelta"] = 0.0

        return df


def generate_realistic_f1_session(
    circuit: str = "Bahrain",
    total_laps: int = 57,
    drivers: list[str] = None,
    sc_laps: Tuple[int, int] = (18, 22),
    vsc_lap: Optional[int] = 34,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates a high-fidelity synthetic F1 race dataset complying with physics:
    - Fuel burn pace gain (~0.035s / lap).
    - Compound baseline differences (Soft is fastest initially, Hard slowest).
    - Non-linear thermal & wear degradation with a tyre cliff.
    - Safety Car & VSC periods reducing pace, freezing wear, and dropping tyre temperature.
    - Pit stops with in-lap / out-lap penalties and pit delta.
    - Track gaps and overtakes.
    """
    if drivers is None:
        drivers = ["VER", "PER", "LEC", "SAI", "HAM", "RUS", "NOR", "PIA", "ALO", "STR"]

    np.random.seed(seed)
    
    # Base lap time for circuit (seconds)
    base_time = 92.5  # ~1:32.500
    fuel_effect = 0.038  # seconds gained per lap as fuel burns (~1.8kg/lap)
    
    # Compound profiles: (initial_pace_advantage, deg_per_lap, cliff_lap, cliff_penalty)
    compounds = {
        "SOFT": {"delta": -0.85, "deg": 0.085, "cliff": 16, "cliff_pen": 0.25},
        "MEDIUM": {"delta": 0.0, "deg": 0.052, "cliff": 26, "cliff_pen": 0.18},
        "HARD": {"delta": 0.65, "deg": 0.032, "cliff": 38, "cliff_pen": 0.12},
    }

    records = []
    
    # Pre-plan default driver strategies (e.g. Medium -> Hard or Soft -> Hard)
    strategies = {
        "VER": [("MEDIUM", 22), ("HARD", 57)],
        "PER": [("MEDIUM", 20), ("HARD", 57)],
        "LEC": [("SOFT", 15), ("HARD", 40), ("SOFT", 57)],
        "SAI": [("MEDIUM", 21), ("HARD", 57)],
        "HAM": [("MEDIUM", 23), ("HARD", 57)],
        "RUS": [("SOFT", 14), ("MEDIUM", 36), ("HARD", 57)],
        "NOR": [("MEDIUM", 24), ("HARD", 57)],
        "PIA": [("MEDIUM", 22), ("HARD", 57)],
        "ALO": [("HARD", 28), ("MEDIUM", 57)],
        "STR": [("MEDIUM", 19), ("HARD", 57)],
    }

    # Weather evolution: track temp drops slightly as evening sets in
    start_track_temp = 36.0

    for driver_idx, driver in enumerate(drivers):
        stint_plan = strategies.get(driver, [("MEDIUM", 22), ("HARD", total_laps)])
        current_stint_idx = 0
        current_compound, stint_end_lap = stint_plan[current_stint_idx]
        tyre_age = 0
        effective_wear = 0.0
        fuel = 105.0
        driver_skill_offset = driver_idx * 0.06 + np.random.normal(0, 0.05)

        for lap in range(1, total_laps + 1):
            tyre_age += 1
            fuel -= 1.84

            track_temp = max(26.0, start_track_temp - (lap * 0.15) + np.random.normal(0, 0.2))
            
            # Event conditions
            is_sc = (sc_laps[0] <= lap <= sc_laps[1])
            is_vsc = (vsc_lap is not None and lap == vsc_lap)
            is_in_lap = False
            is_out_lap = (tyre_age == 1 and lap > 1)

            # Check planned pit stop
            if lap == stint_end_lap and lap < total_laps:
                is_in_lap = True

            # Calculate tyre wear
            if is_sc:
                track_status = "4"
                effective_wear += 0.15
            elif is_vsc:
                track_status = "6"
                effective_wear += 0.35
            else:
                track_status = "1"
                effective_wear += 1.0

            # Calculate Lap Time
            comp_info = compounds[current_compound]
            
            # 1. Base + fuel effect
            fuel_saving = (total_laps - lap) * fuel_effect
            
            # 2. Tyre degradation curve
            wear_deg = effective_wear * comp_info["deg"]
            if effective_wear > comp_info["cliff"]:
                overshoot = effective_wear - comp_info["cliff"]
                wear_deg += (overshoot ** 1.6) * comp_info["cliff_pen"]

            # Temperature sensitivity (hotter track degrades Soft much faster)
            temp_penalty = max(0.0, (track_temp - 30.0) * 0.03 if current_compound == "SOFT" else 0.01)

            lap_time = base_time + driver_skill_offset + comp_info["delta"] + wear_deg + temp_penalty - fuel_saving

            # Apply event penalties
            if is_sc:
                lap_time = base_time * 1.38 + np.random.uniform(0.5, 2.0)
            elif is_vsc:
                lap_time = base_time * 1.25 + np.random.uniform(0.3, 1.0)
            elif is_in_lap:
                lap_time += 3.8  # In-lap pit entry slowdown
            elif is_out_lap:
                lap_time += 22.5 + (1.2 if current_compound == "HARD" else 0.4)  # Pit delta + cold tyre warm-up

            # Normal race variance
            if not is_sc and not is_vsc and not is_in_lap and not is_out_lap:
                lap_time += np.random.normal(0, 0.15)

            records.append({
                "Year": 2024,
                "Race": circuit,
                "Driver": driver,
                "LapNumber": lap,
                "LapTimeSeconds": round(lap_time, 3),
                "Compound": current_compound,
                "TyreLife": tyre_age,
                "EffectiveTyreLife": round(effective_wear, 2),
                "Stint": current_stint_idx + 1,
                "TrackStatus": track_status,
                "IsSafetyCar": is_sc,
                "IsVSC": is_vsc,
                "IsYellow": False,
                "IsOutLap": is_out_lap,
                "IsInLap": is_in_lap,
                "TrackTemp": round(track_temp, 1),
                "AirTemp": 24.5,
                "FuelRemainingKg": round(max(2.0, fuel), 1),
            })

            # Handle Pit Exit to new stint
            if is_in_lap and current_stint_idx + 1 < len(stint_plan):
                current_stint_idx += 1
                current_compound, stint_end_lap = stint_plan[current_stint_idx]
                tyre_age = 0
                effective_wear = 0.0

    df = pd.DataFrame(records)
    
    # Calculate Race Cumulative Time and Track Positions per lap
    df["CumulativeTime"] = df.groupby("Driver")["LapTimeSeconds"].cumsum()
    
    # Assign Position and Gaps
    df["Position"] = df.groupby("LapNumber")["CumulativeTime"].rank(method="first").astype(int)
    
    df = df.sort_values(["LapNumber", "Position"]).reset_index(drop=True)
    
    # Compute gaps
    leader_time = df.groupby("LapNumber")["CumulativeTime"].transform("min")
    df["IntervalToLeader"] = (df["CumulativeTime"] - leader_time).round(3)
    
    df["GapAhead"] = df.groupby("LapNumber")["CumulativeTime"].diff().fillna(0.0).round(3)
    df["GapBehind"] = (-df.groupby("LapNumber")["CumulativeTime"].diff(-1)).fillna(99.0).round(3)

    # Compute baseline and degradation delta
    clean_mask = (~df["IsSafetyCar"]) & (~df["IsVSC"]) & (~df["IsOutLap"]) & (~df["IsInLap"])
    min_clean = df[clean_mask].groupby(["Driver", "Compound"])["LapTimeSeconds"].transform("min")
    df["BaselineLapTime"] = min_clean
    df["DegradationDelta"] = (df["LapTimeSeconds"] - df["BaselineLapTime"]).fillna(0.0).clip(lower=0.0).round(3)

    return df
