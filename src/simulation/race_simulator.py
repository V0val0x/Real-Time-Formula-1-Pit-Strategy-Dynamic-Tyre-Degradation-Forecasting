"""
F1 Live Race Simulator and Strategy Replay Engine.
Replays a race lap-by-lap, updates live telemetry and competitor states,
and demonstrates real-time dynamic strategy branching and pivots.
"""

from __future__ import annotations
import time
from typing import Dict, List, Optional, Callable
import pandas as pd
from ..data.f1_data import generate_realistic_f1_session
from ..models.tyre_model import TyreDegradationModel
from ..strategy.strategy_engine import StrategyRecommender, RaceState, CompetitorInfo, StrategyDecision


class RaceSimulator:
    """
    Simulates or replays a Formula 1 race lap-by-lap, feeding live data
    into the ML Degradation model and Strategy Recommender.
    """

    def __init__(
        self,
        race_df: Optional[pd.DataFrame] = None,
        target_driver: str = "VER",
        model: Optional[TyreDegradationModel] = None,
    ):
        if race_df is None:
            race_df = generate_realistic_f1_session(circuit="Bahrain", total_laps=57, sc_laps=(18, 21), vsc_lap=34)
        self.race_df = race_df
        self.target_driver = target_driver
        self.total_laps = int(race_df["LapNumber"].max())

        # Fit model on baseline data
        if model is None:
            self.model = TyreDegradationModel()
            self.model.fit(self.race_df)
        else:
            self.model = model

        self.recommender = StrategyRecommender(self.model)
        self.history: List[Dict] = []

    def run_replay(
        self,
        on_lap_callback: Optional[Callable[[StrategyDecision, RaceState], None]] = None,
        delay_seconds: float = 0.0,
    ) -> pd.DataFrame:
        """
        Iterates through the race lap-by-lap.
        At each lap, constructs the live state, queries the strategy engine,
        and records decisions and contingency branches.
        """
        driver_df = self.race_df[self.race_df["Driver"] == self.target_driver].sort_values("LapNumber")
        
        recent_laps_window: List[float] = []

        for lap_no in range(1, self.total_laps + 1):
            lap_row = driver_df[driver_df["LapNumber"] == lap_no]
            if lap_row.empty:
                continue
            lap_data = lap_row.iloc[0]

            lap_time = float(lap_data["LapTimeSeconds"])
            recent_laps_window.append(lap_time)
            if len(recent_laps_window) > 5:
                recent_laps_window.pop(0)

            # Build competitor states for this lap
            all_lap_df = self.race_df[self.race_df["LapNumber"] == lap_no]
            target_cum_time = float(lap_data["CumulativeTime"]) if "CumulativeTime" in lap_data else lap_time * lap_no

            competitors: List[CompetitorInfo] = []
            for _, rival in all_lap_df.iterrows():
                r_driver = rival["Driver"]
                if r_driver == self.target_driver:
                    continue
                r_cum_time = float(rival["CumulativeTime"]) if "CumulativeTime" in rival else rival["LapTimeSeconds"] * lap_no
                # interval: positive if rival is ahead, negative if rival is behind
                interval = round(target_cum_time - r_cum_time, 2)
                
                competitors.append(CompetitorInfo(
                    driver=r_driver,
                    position=int(rival["Position"]),
                    compound=str(rival["Compound"]),
                    tyre_age=int(rival["TyreLife"]),
                    interval_to_target=interval,
                    has_pitted_recently=bool(rival["IsInLap"] or rival["IsOutLap"]),
                    last_lap_time=float(rival["LapTimeSeconds"]),
                ))

            # Current state
            state = RaceState(
                lap=lap_no,
                total_laps=self.total_laps,
                driver=self.target_driver,
                position=int(lap_data["Position"]),
                compound=str(lap_data["Compound"]),
                tyre_age=int(lap_data["TyreLife"]),
                effective_wear=float(lap_data["EffectiveTyreLife"]),
                current_pace=lap_time,
                recent_laps=list(recent_laps_window),
                track_status=str(lap_data["TrackStatus"]),
                track_temp=float(lap_data["TrackTemp"]),
                fuel_kg=float(lap_data["FuelRemainingKg"]),
                gap_ahead=float(lap_data.get("GapAhead", 0.0)),
                gap_behind=float(lap_data.get("GapBehind", 99.0)),
                is_before_pit_entry=True,  # Assuming decision before pit entry commitment line
                competitors=competitors,
            )

            # Query real-time strategy recommender
            decision = self.recommender.evaluate_live_lap(state)

            record = {
                "LapNumber": lap_no,
                "Position": state.position,
                "Compound": state.compound,
                "TyreLife": state.tyre_age,
                "EffectiveTyreLife": state.effective_wear,
                "TrackStatus": state.track_status,
                "LapTimeSeconds": state.current_pace,
                "ActivePlan": decision.active_plan,
                "PrimaryAction": decision.primary_action,
                "TargetCompound": decision.target_compound,
                "LapsToCliff": decision.tyre_cliff_prediction[0],
                "CliffPaceDrop": decision.tyre_cliff_prediction[1],
                "Explanation": decision.explanation,
                "NumContingencyBranches": len(decision.contingency_branches),
                "TacticalAlerts": "; ".join(decision.tactical_alerts) if decision.tactical_alerts else "None",
            }
            # Add specific branch actions
            for branch in decision.contingency_branches:
                record[f"Branch_{branch.branch_name}"] = branch.recommended_action

            self.history.append(record)

            if on_lap_callback:
                on_lap_callback(decision, state)

            if delay_seconds > 0:
                time.sleep(delay_seconds)

        return pd.DataFrame(self.history)
