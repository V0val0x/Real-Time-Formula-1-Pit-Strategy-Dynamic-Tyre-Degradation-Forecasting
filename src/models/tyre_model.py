"""
F1 Tyre Degradation Machine Learning Model.
Trains on race data to forecast lap degradation delta, detect tyre cliffs,
and evaluate compound life expectancy under variable track conditions.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except (ImportError, OSError):
    HAS_LIGHTGBM = False

from sklearn.ensemble import GradientBoostingRegressor


class TyreDegradationModel:
    """
    Predicts tyre degradation delta (seconds lost compared to fresh tyre base pace).
    Includes physical cliff detection and scenario simulations.
    """

    FEATURE_COLS = [
        "Compound_Code",        # 0: Hard, 1: Medium, 2: Soft
        "TyreLife",             # Nominal laps on set
        "EffectiveTyreLife",    # SC-adjusted wear laps
        "TrackTemp",            # Track surface temperature (°C)
        "FuelRemainingKg",      # Current estimated fuel mass
        "IsVerstappen",         # 1: VER, 0: PER
        "IsSafetyCar",          # 0 or 1
        "IsVSC",                # 0 or 1
        "IsOutLap",             # 0 or 1
    ]

    COMPOUND_MAP = {"HARD": 0, "MEDIUM": 1, "SOFT": 2}
    INV_COMPOUND_MAP = {0: "HARD", 1: "MEDIUM", 2: "SOFT"}

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        if HAS_LIGHTGBM:
            self.model = lgb.LGBMRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=24,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=random_state,
                verbose=-1,
            )
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.08,
                max_depth=4,
                random_state=random_state,
            )
        self.is_fitted = False
        self.baseline_times: Dict[str, float] = {}

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms raw dataframe columns into numerical features."""
        data = df.copy()
        if "Compound_Code" not in data.columns:
            data["Compound_Code"] = data["Compound"].astype(str).str.upper().map(self.COMPOUND_MAP).fillna(1)
        
        if "IsVerstappen" not in data.columns:
            if "Driver" in data.columns:
                data["IsVerstappen"] = (data["Driver"].astype(str).str.upper() == "VER").astype(int)
            else:
                data["IsVerstappen"] = 1

        for col in ["IsSafetyCar", "IsVSC", "IsOutLap", "IsVerstappen"]:
            if col in data.columns:
                data[col] = data[col].astype(int)
            else:
                data[col] = 0

        # Ensure all columns exist
        for col in self.FEATURE_COLS:
            if col not in data.columns:
                data[col] = 0.0

        return data[self.FEATURE_COLS]

    def fit(self, train_df: pd.DataFrame, target_col: str = "DegradationDelta") -> Dict[str, float]:
        """Trains the degradation model on historical or current session laps."""
        # Cache baseline times per compound
        clean_mask = (~train_df["IsSafetyCar"]) & (~train_df["IsVSC"]) & (~train_df["IsOutLap"]) & (~train_df["IsInLap"])
        clean_df = train_df[clean_mask]
        for comp in ["SOFT", "MEDIUM", "HARD"]:
            comp_laps = clean_df[clean_df["Compound"] == comp]["LapTimeSeconds"]
            self.baseline_times[comp] = float(comp_laps.quantile(0.05)) if not comp_laps.empty else 92.0

        X = self._prepare_features(train_df)
        y = train_df[target_col].values

        self.model.fit(X, y)
        self.is_fitted = True

        preds = self.model.predict(X)
        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y, preds))),
            "mae": float(mean_absolute_error(y, preds)),
            "r2": float(r2_score(y, preds)),
        }
        return metrics

    def predict_delta(
        self,
        compound: str,
        tyre_life: float,
        effective_tyre_life: float,
        track_temp: float,
        fuel_remaining_kg: float,
        driver: str = "VER",
        is_sc: bool = False,
        is_vsc: bool = False,
        is_out_lap: bool = False,
    ) -> float:
        """Predicts expected lap degradation penalty in seconds."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict.")

        is_ver = 1 if str(driver).upper() == "VER" else 0

        feat_row = pd.DataFrame([{
            "Compound_Code": self.COMPOUND_MAP.get(compound.upper(), 1),
            "TyreLife": float(tyre_life),
            "EffectiveTyreLife": float(effective_tyre_life),
            "TrackTemp": float(track_temp),
            "FuelRemainingKg": float(fuel_remaining_kg),
            "IsVerstappen": is_ver,
            "IsSafetyCar": int(is_sc),
            "IsVSC": int(is_vsc),
            "IsOutLap": int(is_out_lap),
        }])[self.FEATURE_COLS]

        pred = self.model.predict(feat_row)[0]
        return float(max(0.0, pred))

    def predict_lap_time(
        self,
        compound: str,
        tyre_life: float,
        effective_tyre_life: float,
        track_temp: float,
        fuel_remaining_kg: float,
        is_sc: bool = False,
        is_vsc: bool = False,
        is_out_lap: bool = False,
    ) -> float:
        """Computes absolute expected lap time based on compound baseline + predicted delta."""
        base = self.baseline_times.get(compound.upper(), 92.0)
        # Fuel burn benefit relative to 105kg: ~0.035s / kg
        fuel_delta = (fuel_remaining_kg - 105.0) * 0.035
        deg = self.predict_delta(
            compound, tyre_life, effective_tyre_life, track_temp, fuel_remaining_kg, is_sc, is_vsc, is_out_lap
        )
        return base + fuel_delta + deg

    def detect_tyre_cliff(
        self,
        compound: str,
        current_effective_life: float,
        track_temp: float,
        fuel_kg: float,
        deg_threshold_per_lap: float = 0.22,
    ) -> Tuple[int, float]:
        """
        Simulates future laps to find the lap where tyre degradation accelerates
        beyond the cliff threshold (> deg_threshold_per_lap seconds drop per lap).
        Returns: (laps_remaining_to_cliff, cliff_pace_penalty)
        """
        eff_life = current_effective_life
        tyre_life = current_effective_life
        curr_fuel = fuel_kg
        prev_deg = self.predict_delta(compound, tyre_life, eff_life, track_temp, curr_fuel)

        for lap_step in range(1, 40):
            tyre_life += 1.0
            eff_life += 1.0
            curr_fuel = max(2.0, curr_fuel - 1.8)
            next_deg = self.predict_delta(compound, tyre_life, eff_life, track_temp, curr_fuel)
            
            lap_loss = next_deg - prev_deg
            if lap_loss >= deg_threshold_per_lap:
                return lap_step, round(next_deg, 3)
            prev_deg = next_deg

        return 35, round(prev_deg, 3)

    def get_feature_importance(self) -> Dict[str, float]:
        """Returns relative feature importances for Explainable AI (XAI)."""
        if not self.is_fitted:
            return {}
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            total = sum(importances) if sum(importances) > 0 else 1.0
            return {col: round(float(imp / total), 4) for col, imp in zip(self.FEATURE_COLS, importances)}
        return {}
