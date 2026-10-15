"""
Real-Time F1 Strategy & Multi-Scenario Contingency Engine.
Evaluates race state, competitor moves, tyre degradation, traffic windows,
and computes dynamic contingency branches (e.g. IF VSC THEN..., IF Pace < X THEN...).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from ..models.tyre_model import TyreDegradationModel
from .pit_dynamics import PitDynamicsModel


@dataclass
class CompetitorInfo:
    driver: str
    position: int
    compound: str
    tyre_age: int
    interval_to_target: float  # positive if ahead, negative if behind
    has_pitted_recently: bool = False
    last_lap_time: float = 93.0


@dataclass
class RaceState:
    lap: int
    total_laps: int
    driver: str
    position: int
    compound: str
    tyre_age: int
    effective_wear: float
    current_pace: float
    recent_laps: List[float]
    track_status: str           # "1": Green, "4": SC, "6": VSC
    track_temp: float
    fuel_kg: float
    gap_ahead: float            # Seconds to car in front
    gap_behind: float           # Seconds to car behind
    is_before_pit_entry: bool = True
    competitors: List[CompetitorInfo] = field(default_factory=list)
    pit_delta_green: float = 22.5
    pit_delta_sc: float = 11.5


@dataclass
class ScenarioBranch:
    branch_name: str            # e.g., "Plan A (Default)", "VSC Contingency", "Pace Cliff Trigger"
    condition_description: str  # e.g., "If VSC occurs AND gap behind > 12.0s"
    recommended_action: str     # "BOX_THIS_LAP", "STAY_OUT", "BOX_IN_LAPS(N)"
    target_compound: str        # "HARD", "MEDIUM", "SOFT"
    rejoin_position: int        # Predicted re-entry position
    clean_air_predicted: bool   # Re-enters into clean air vs traffic train
    expected_race_time_delta: float  # Net seconds saved/lost relative to baseline
    rationale: str              # Domain tactical rationale


@dataclass
class StrategyDecision:
    current_lap: int
    active_plan: str
    primary_action: str         # "BOX_THIS_LAP", "STAY_OUT", "PREPARE_BOX"
    target_compound: str
    contingency_branches: List[ScenarioBranch]
    tyre_cliff_prediction: Tuple[int, float]  # (laps_remaining_to_cliff, expected_drop)
    tactical_alerts: List[str]  # e.g., "Undercut threat from HAM", "Traffic window clear"
    explanation: str


class StrategyRecommender:
    """
    Real-time tactical decision engine for Formula 1 pit strategy.
    Continuously monitors telemetry, competitors, degradation curves, and
    evaluates multiple 'What-If' scenarios concurrently.
    """

    def __init__(self, model: TyreDegradationModel):
        self.model = model
        self.pit_model = PitDynamicsModel(team="Red Bull")
        self.active_strategy_name = "Plan A (Optimum One-Stop)"
        self.planned_pit_lap = 24
        self.secondary_planned_compound = "HARD"

    def evaluate_live_lap(self, state: RaceState) -> StrategyDecision:
        """
        Main decision-making function executed every lap (or in real-time replay).
        Analyzes current status and generates primary recommendation + contingency branches.
        """
        laps_remaining = state.total_laps - state.lap
        alerts: List[str] = []
        branches: List[ScenarioBranch] = []

        # 1. Evaluate Tyre Degradation & Cliff
        laps_to_cliff, cliff_pace_penalty = self.model.detect_tyre_cliff(
            compound=state.compound,
            current_effective_life=state.effective_wear,
            track_temp=state.track_temp,
            fuel_kg=state.fuel_kg,
        )

        cliff_lap_estimate = state.lap + laps_to_cliff

        # Check for early tyre cliff or severe pace degradation
        is_cliff_imminent = (laps_to_cliff <= 2)
        degradation_trend = self._calc_degradation_trend(state.recent_laps)
        
        # 2. Competitor Tactical Analysis (Undercut / Overcut / Traffic)
        undercut_threat = self._check_undercut_threat(state)
        if undercut_threat:
            alerts.append(f"⚠️ UNDERCUT THREAT: {undercut_threat.driver} pitted recently and is closing delta!")

        # 3. Strategy Targets
        target_compound = "HARD" if state.compound in ["SOFT", "MEDIUM"] else "MEDIUM"
        traffic_rejoin_pos, is_clean_air = self._predict_pit_rejoin(
            state, is_sc_active=state.track_status in ["4", "6"], target_compound=target_compound
        )
        
        # Branch 1: Baseline Strategy (Plan A)
        if state.lap < self.planned_pit_lap - 2:
            plan_a_action = f"STAY_OUT (Target Box: Lap {self.planned_pit_lap})"
        elif state.lap == self.planned_pit_lap:
            plan_a_action = "BOX_THIS_LAP"
        else:
            plan_a_action = "BOX_WINDOW_OPEN"

        branches.append(ScenarioBranch(
            branch_name="Plan A (Nominal Strategy)",
            condition_description=f"Normal green flag running, pace degradation stable (+{degradation_trend:.2f}s/lap)",
            recommended_action=plan_a_action,
            target_compound=target_compound,
            rejoin_position=traffic_rejoin_pos,
            clean_air_predicted=is_clean_air,
            expected_race_time_delta=0.0,
            rationale=f"Maximizes stint balance on {state.compound} -> {target_compound}. Expected cliff around Lap {cliff_lap_estimate}."
        ))

        # 4. Branch 2: Safety Car / VSC Contingency
        vsc_branch = self._evaluate_vsc_contingency(state, target_compound)
        branches.append(vsc_branch)

        # 5. Branch 3: Pace Threshold / Early Cliff Trigger
        pace_threshold = 94.2  # lap time cliff trigger threshold
        early_cliff_branch = self._evaluate_pace_trigger(state, degradation_trend, target_compound, pace_threshold)
        branches.append(early_cliff_branch)

        # 6. Branch 4: High Pace Preservation / Overcut Stint Extension
        extension_branch = self._evaluate_extension_trigger(state, degradation_trend, target_compound)
        branches.append(extension_branch)

        # 7. Dynamic Strategy Resolution: Determine Active Primary Action
        primary_action, active_plan, explanation = self._resolve_primary_action(
            state=state,
            branches=branches,
            is_cliff_imminent=is_cliff_imminent,
            undercut_threat=undercut_threat,
            is_clean_air=is_clean_air,
        )

        return StrategyDecision(
            current_lap=state.lap,
            active_plan=active_plan,
            primary_action=primary_action,
            target_compound=target_compound,
            contingency_branches=branches,
            tyre_cliff_prediction=(laps_to_cliff, cliff_pace_penalty),
            tactical_alerts=alerts,
            explanation=explanation,
        )

    def _calc_degradation_trend(self, recent_laps: List[float]) -> float:
        """Calculates pace slope over recent 3-5 laps."""
        if len(recent_laps) < 2:
            return 0.05
        diffs = [recent_laps[i] - recent_laps[i-1] for i in range(1, len(recent_laps))]
        # filter out pit/sc anomalies
        valid_diffs = [d for d in diffs if abs(d) < 3.0]
        return float(np.mean(valid_diffs)) if valid_diffs else 0.05

    def _check_undercut_threat(self, state: RaceState) -> Optional[CompetitorInfo]:
        """Checks if a close competitor behind (< 2.5s) pitted in the last 1-2 laps."""
        for comp in state.competitors:
            if comp.interval_to_target < 0 and abs(comp.interval_to_target) <= 2.8:
                if comp.has_pitted_recently:
                    return comp
        return None

    def _predict_pit_rejoin(
        self, state: RaceState, is_sc_active: bool, target_compound: str = "HARD"
    ) -> Tuple[int, bool]:
        """
        Simulates track position after pit exit.
        Takes into account pit transit loss and compound cold tyre warm-up penalty.
        Returns: (predicted_position, is_clean_air_ahead)
        """
        base_pit_loss = state.pit_delta_sc if is_sc_active else state.pit_delta_green
        cold_pen, _ = self.pit_model.calculate_cold_tyre_penalty(target_compound, state.track_temp)
        pit_loss = base_pit_loss + cold_pen
        rejoin_pos = state.position

        # Check which competitors overtake during pit stop
        cars_ahead_on_rejoin = []
        for comp in state.competitors:
            if comp.interval_to_target < 0:  # was behind us
                # If they were closer than pit_loss seconds, they pass us
                if abs(comp.interval_to_target) < pit_loss:
                    rejoin_pos += 1
                    cars_ahead_on_rejoin.append(comp)

        # Check if re-entering directly behind someone (< 1.5s)
        is_clean_air = True
        if cars_ahead_on_rejoin:
            # Car directly ahead after pit exit
            immediate_ahead = cars_ahead_on_rejoin[-1]
            estimated_gap = pit_loss - abs(immediate_ahead.interval_to_target)
            if estimated_gap < 1.5:
                is_clean_air = False

        return min(len(state.competitors) + 1, rejoin_pos), is_clean_air

    def _evaluate_vsc_contingency(self, state: RaceState, target_compound: str) -> ScenarioBranch:
        """
        Calculates non-trivial VSC/SC opportunity:
        - Checks position on track (is_before_pit_entry).
        - Evaluates tyre age (is it worth pitting, or are tyres fresh?).
        - Evaluates traffic window.
        """
        sc_rejoin_pos, sc_clean_air = self._predict_pit_rejoin(state, is_sc_active=True)
        time_saved_vs_green = state.pit_delta_green - state.pit_delta_sc  # ~11 seconds advantage

        # Tactical condition:
        # Pit under VSC/SC ONLY IF:
        # 1. Car is before pit entry (can physically enter pit lane this lap)
        # 2. Tyres have accumulated at least 7 effective laps (otherwise waste of fresh rubber)
        # 3. Won't re-emerge in a severe DRS traffic train
        if not state.is_before_pit_entry:
            action = "STAY_OUT (Missed pit entry this lap, prepare next lap if SC continues)"
            rationale = "Car was past pit entry line when SC/VSC was declared."
        elif state.effective_wear < 7.0:
            action = "STAY_OUT (Tyres too fresh to discard)"
            rationale = f"Tyres only {state.effective_wear:.1f} laps old. Track position is more valuable than cheap stop."
        elif not sc_clean_air and sc_rejoin_pos > 8:
            action = "WEIGH_TRAFFIC_RISK (Rejoins in DRS train P{sc_rejoin_pos})"
            rationale = f"Saves {time_saved_vs_green:.1f}s, but high risk of traffic congestion on hard compound."
        else:
            action = "BOX_IMMEDIATELY_UNDER_SC"
            rationale = f"Optimal cheap pit stop! Net time saved ~{time_saved_vs_green:.1f}s, rejoins P{sc_rejoin_pos} in clean air."

        return ScenarioBranch(
            branch_name="VSC / Safety Car Contingency",
            condition_description="IF TrackStatus changes to SC ('4') or VSC ('6') AND car is before pit entry",
            recommended_action=action,
            target_compound=target_compound,
            rejoin_position=sc_rejoin_pos,
            clean_air_predicted=sc_clean_air,
            expected_race_time_delta=-time_saved_vs_green if "BOX" in action else 0.0,
            rationale=rationale
        )

    def _evaluate_pace_trigger(
        self, state: RaceState, trend: float, target_compound: str, threshold_seconds: float
    ) -> ScenarioBranch:
        """Contingency if tyre degradation accelerates prematurely (cliff arrived early)."""
        rejoin_pos, clean_air = self._predict_pit_rejoin(state, is_sc_active=False)
        return ScenarioBranch(
            branch_name="Premature Tyre Cliff Trigger",
            condition_description=f"IF LapTime > {threshold_seconds:.2f}s OR consecutive pace drop > 0.40s/lap",
            recommended_action=f"BOX_IN_LAPS(1) (Emergency pit stop on Lap {state.lap + 1})",
            target_compound=target_compound,
            rejoin_position=rejoin_pos,
            clean_air_predicted=clean_air,
            expected_race_time_delta=-4.5,  # Saves ~4.5s compared to losing 1.5s/lap on dead tyres
            rationale=f"Mitigates severe pace bleed. Prevents dropping out of undercut window."
        )

    def _evaluate_extension_trigger(
        self, state: RaceState, trend: float, target_compound: str
    ) -> ScenarioBranch:
        """Contingency if tyre preservation is superior to expectations (overcut opportunity)."""
        extended_lap = min(state.total_laps - 12, self.planned_pit_lap + 7)
        return ScenarioBranch(
            branch_name="Stint Extension / Overcut Trigger",
            condition_description=f"IF pace degradation stays < 0.035s/lap AND gap ahead < 2.0s",
            recommended_action=f"EXTEND_STINT (New Target Box: Lap {extended_lap})",
            target_compound=target_compound,
            rejoin_position=state.position,
            clean_air_predicted=True,
            expected_race_time_delta=-2.8,  # Net gain from overcut and fresher final stint tyres
            rationale=f"Superb tyre management enables overcut. Guarantees fresher {target_compound} for the finish."
        )

    def _resolve_primary_action(
        self,
        state: RaceState,
        branches: List[ScenarioBranch],
        is_cliff_imminent: bool,
        undercut_threat: Optional[CompetitorInfo],
        is_clean_air: bool,
    ) -> Tuple[str, str, str]:
        """Dynamically decides which branch should be activated right now."""
        # Condition A: Active SC / VSC
        if state.track_status in ["4", "6"]:
            vsc_branch = branches[1]
            if "BOX_IMMEDIATELY" in vsc_branch.recommended_action:
                return (
                    "BOX_THIS_LAP",
                    "Plan VSC (Opportunistic Cheap Stop)",
                    f"VSC/SC active. Exploiting cheap pit delta ({state.pit_delta_sc}s vs {state.pit_delta_green}s). {vsc_branch.rationale}"
                )

        # Condition B: Covering Undercut Threat
        if undercut_threat and state.is_before_pit_entry and state.lap >= self.planned_pit_lap - 4:
            return (
                "BOX_THIS_LAP",
                "Plan U (Undercut Defense)",
                f"Urgent response: Rival {undercut_threat.driver} is undercutting. Pitting immediately to protect track position."
            )

        # Condition C: Imminent Tyre Cliff
        if is_cliff_imminent and state.is_before_pit_entry and state.lap >= 14:
            return (
                "BOX_THIS_LAP",
                "Plan C (Cliff Mitigation)",
                f"Tyres reached critical degradation limit. Expected cliff penalty is severe (+{branches[2].expected_race_time_delta:.1f}s if delayed)."
            )

        # Check if already completed planned stop (e.g. tyre age is smaller than current lap by a margin)
        has_completed_stop = (state.lap > self.planned_pit_lap and state.tyre_age < state.lap - 4)

        # Condition D: Reached planned window (only if haven't pitted yet)
        if not has_completed_stop and state.lap >= self.planned_pit_lap:
            return (
                "BOX_THIS_LAP",
                "Plan A (Nominal Target Reached)",
                f"Scheduled pit window reached on Lap {state.lap}. Rejoining in predicted P{branches[0].rejoin_position}."
            )

        # Stint 2 / Final Stint in progress
        if has_completed_stop:
            laps_to_finish = state.total_laps - state.lap
            return (
                "STAY_OUT",
                "Plan A (Stint 2 to Finish)",
                f"Running on target final {state.compound} stint. {laps_to_finish} laps to finish. Monitoring tyre wear."
            )

        # Default: Stay out on current stint
        laps_to_go = self.planned_pit_lap - state.lap
        return (
            "STAY_OUT",
            "Plan A (Stint Management)",
            f"Nominal pace is optimal. {laps_to_go} laps until scheduled pit stop. Monitoring VSC and undercut branches."
        )
