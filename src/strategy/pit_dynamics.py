"""
F1 Pit Stop Dynamics, In/Out-Lap Physics, and Cold Tyre Warm-up Modeling.
Models the complete pit cycle:
1. In-lap deceleration and pit entry delta.
2. Stationary wheel change time (Red Bull standard ~2.0-2.3s).
3. Pit lane transit time under Green vs SC/VSC.
4. Out-lap cold tyre penalty as a non-linear function of compound and track temperature.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np


@dataclass
class TrackPitProfile:
    circuit: str
    pit_lane_delta_green: float     # Total pit lane transit + box time under green (seconds)
    pit_lane_delta_sc: float        # Total pit lane transit + box time under SC/VSC (seconds)
    in_lap_decel_loss: float        # In-lap time lost approaching pit entry (seconds)
    pit_lane_speed_limit_kph: int   # 60 or 80 km/h


# Circuit-specific pit profiles based on official FIA timing
TRACK_PIT_PROFILES: Dict[str, TrackPitProfile] = {
    "Bahrain": TrackPitProfile(
        circuit="Bahrain",
        pit_lane_delta_green=23.5,
        pit_lane_delta_sc=11.8,
        in_lap_decel_loss=4.8,
        pit_lane_speed_limit_kph=80,
    ),
    "Spain": TrackPitProfile(
        circuit="Spain",
        pit_lane_delta_green=21.8,
        pit_lane_delta_sc=11.2,
        in_lap_decel_loss=4.2,
        pit_lane_speed_limit_kph=80,
    ),
    "Great Britain": TrackPitProfile(
        circuit="Great Britain",
        pit_lane_delta_green=24.5,
        pit_lane_delta_sc=12.5,
        in_lap_decel_loss=5.0,
        pit_lane_speed_limit_kph=80,
    ),
    "Monza": TrackPitProfile(
        circuit="Monza",
        pit_lane_delta_green=24.2,
        pit_lane_delta_sc=12.0,
        in_lap_decel_loss=4.5,
        pit_lane_speed_limit_kph=80,
    ),
    "Monaco": TrackPitProfile(
        circuit="Monaco",
        pit_lane_delta_green=25.8,
        pit_lane_delta_sc=14.2,
        in_lap_decel_loss=3.8,
        pit_lane_speed_limit_kph=60,
    ),
    "Austria": TrackPitProfile(
        circuit="Austria",
        pit_lane_delta_green=20.2,
        pit_lane_delta_sc=10.4,
        in_lap_decel_loss=3.5,
        pit_lane_speed_limit_kph=80,
    ),
    "Japan": TrackPitProfile(
        circuit="Japan",
        pit_lane_delta_green=22.8,
        pit_lane_delta_sc=11.5,
        in_lap_decel_loss=4.6,
        pit_lane_speed_limit_kph=80,
    ),
}

DEFAULT_PROFILE = TrackPitProfile(
    circuit="Default",
    pit_lane_delta_green=22.5,
    pit_lane_delta_sc=11.5,
    in_lap_decel_loss=4.0,
    pit_lane_speed_limit_kph=80,
)


class PitDynamicsModel:
    """
    Computes precise time losses for pit in-laps, pit stops, and out-lap warm-up.
    """

    def __init__(self, team: str = "Red Bull"):
        self.team = team
        # Red Bull baseline stationary pit stop time
        self.base_stationary_stop = 2.15  # seconds
        self.stop_variance = 0.20

    def get_track_profile(self, circuit: str) -> TrackPitProfile:
        return TRACK_PIT_PROFILES.get(circuit, DEFAULT_PROFILE)

    def calculate_cold_tyre_penalty(self, compound: str, track_temp: float) -> Tuple[float, float]:
        """
        Calculates out-lap and lap+1 warm-up penalty (seconds).
        With 70°C blankets, tyres must be brought up to ~105°C operating window.
        Hard tyres suffer most, especially on cold tracks (< 28°C).
        Returns: (out_lap_penalty, second_lap_penalty)
        """
        comp = compound.upper()
        # Temperature deficit factor: colder track -> steeper penalty
        temp_deficit = max(0.0, 32.0 - track_temp)

        if comp == "HARD":
            # Hard compound (C1/C2) is notoriously cold-sensitive
            out_lap_pen = 1.95 + (temp_deficit * 0.065)
            second_lap_pen = 0.55 + (temp_deficit * 0.025)
        elif comp == "MEDIUM":
            # Medium (C3) standard warm-up
            out_lap_pen = 1.05 + (temp_deficit * 0.040)
            second_lap_pen = 0.20 + (temp_deficit * 0.010)
        elif comp == "SOFT":
            # Soft (C4/C5) comes into window quickly, minimal warm-up loss
            out_lap_pen = 0.45 + (temp_deficit * 0.020)
            second_lap_pen = 0.00
        else:
            out_lap_pen = 1.20
            second_lap_pen = 0.20

        return round(out_lap_pen, 3), round(second_lap_pen, 3)

    def calculate_total_pit_cycle_loss(
        self,
        circuit: str,
        compound_new: str,
        track_temp: float,
        is_sc: bool = False,
    ) -> Dict[str, float]:
        """
        Calculates the complete time penalty of pitting:
        Total Loss = (Transit Delta) + (In-Lap Loss) + (Cold Tyre Penalty)
        """
        profile = self.get_track_profile(circuit)
        transit_loss = profile.pit_lane_delta_sc if is_sc else profile.pit_lane_delta_green
        in_lap_loss = profile.in_lap_decel_loss
        
        # Out-lap cold tyre penalty
        cold_pen_out, cold_pen_lap2 = self.calculate_cold_tyre_penalty(compound_new, track_temp)
        
        # Total time loss relative to staying out on clean baseline pace
        net_cycle_loss = transit_loss + in_lap_loss + cold_pen_out

        return {
            "transit_loss": round(transit_loss, 2),
            "in_lap_loss": round(in_lap_loss, 2),
            "cold_tyre_outlap_penalty": round(cold_pen_out, 2),
            "cold_tyre_lap2_penalty": round(cold_pen_lap2, 2),
            "total_pit_cycle_loss": round(net_cycle_loss, 2),
            "is_sc": is_sc,
        }

    def evaluate_undercut_opportunity(
        self,
        gap_to_target_ahead: float,
        rival_pace_loss_per_lap: float,
        fresh_compound: str,
        track_temp: float,
        circuit: str,
    ) -> Dict[str, Any]:
        """
        Evaluates whether an undercut will succeed.
        Undercut works if the fresh tyre pace advantage over 1-2 laps
        exceeds the rival's time on their worn tyres.
        """
        cold_pen, _ = self.calculate_cold_tyre_penalty(fresh_compound, track_temp)
        
        # Fresh tyre pace advantage over worn tyre (e.g. fresh Medium vs 25-lap worn Medium)
        pace_advantage_clean = 1.85  # seconds faster on fresh rubber
        effective_outlap_gain = pace_advantage_clean - cold_pen

        # If gap is smaller than the delta gained over the undercut window
        max_undercut_gap = effective_outlap_gain + pace_advantage_clean
        is_undercut_viable = (gap_to_target_ahead <= max_undercut_gap)

        return {
            "is_viable": is_undercut_viable,
            "current_gap": round(gap_to_target_ahead, 2),
            "max_catchable_gap": round(max_undercut_gap, 2),
            "cold_tyre_drag": round(cold_pen, 2),
            "net_outlap_advantage": round(effective_outlap_gain, 2),
            "recommendation": "BOX_NOW_FOR_UNDERCUT" if is_undercut_viable else "STAY_OUT_GAP_TOO_LARGE",
        }
