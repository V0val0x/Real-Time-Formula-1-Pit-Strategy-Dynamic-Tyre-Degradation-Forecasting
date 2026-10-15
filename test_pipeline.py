import os
import joblib
import pandas as pd
from src.data.f1_data import generate_realistic_f1_session
from src.models.tyre_model import TyreDegradationModel
from src.simulation.race_simulator import RaceSimulator


def main():
    print("=" * 70)
    print("🏁 INITIALIZING F1 TYRE DEGRADATION & REAL-TIME STRATEGY ENGINE 🏁")
    print("=" * 70)

    # 1. Load Real or Synthetic Race Session
    data_path = "data/processed/f1_curated_telemetry.csv"
    if os.path.exists(data_path):
        print("\n[1/3] Loading real 2023 Bahrain GP session for Red Bull Racing...")
        all_df = pd.read_csv(data_path)
        race_df = all_df[(all_df["Year"] == 2023) & (all_df["Race"] == "Bahrain")].copy()
        if race_df.empty:
            race_df = generate_realistic_f1_session(circuit="Bahrain", total_laps=57, sc_laps=(18, 21), vsc_lap=34)
    else:
        print("\n[1/3] Generating realistic Bahrain GP race session (57 laps)...")
        race_df = generate_realistic_f1_session(circuit="Bahrain", total_laps=57, sc_laps=(18, 21), vsc_lap=34)

    print(f"✓ Total race records loaded: {len(race_df)} laps across {race_df['Driver'].nunique()} drivers.")

    # 2. Load or Train Tyre Degradation Model
    model_path = "src/models/trained_tyre_model.joblib"
    if os.path.exists(model_path):
        print("\n[2/3] Loading Pre-Trained Model from src/models/trained_tyre_model.joblib...")
        model = joblib.load(model_path)
    else:
        print("\n[2/3] Training Physics-Aware Tyre Degradation Model...")
        model = TyreDegradationModel()
        model.fit(race_df)

    print("✓ Feature Importances:")
    for feat, imp in model.get_feature_importance().items():
        print(f"   • {feat:<20}: {imp*100:.1f}%")

    # 3. Run Live Race Replay for VER
    print("\n[3/3] Running Live Race Replay for Target Driver: VER (Max Verstappen)...")
    print("-" * 70)

    def print_lap_event(decision, state):
        interesting_laps = [1, 10, 14, 15, 18, 19, 24, 34, 36, 40, 50, 57]
        is_interesting = (state.lap in interesting_laps) or ("BOX" in decision.primary_action)
        if is_interesting:
            status_tag = "🟩 GREEN" if state.track_status == "1" else ("🟨 SC ACTIVE" if "4" in str(state.track_status) else "🟨 VSC")
            print(f"\n📍 [LAP {state.lap:02d}/57] Pos: P{state.position} | Tyre: {state.compound} (Age: {state.tyre_age}, EffWear: {state.effective_wear:.1f}) | Status: {status_tag}")
            print(f"   ⏱ Pace: {state.current_pace:.3f}s | Cliff in: ~{decision.tyre_cliff_prediction[0]} laps | Fuel: {state.fuel_kg:.1f}kg")
            print(f"   🎯 Active Plan: {decision.active_plan}")
            print(f"   📢 Primary Action: >>> {decision.primary_action} <<<")
            print(f"   💡 Strategy Rationale: {decision.explanation}")
            
            print("   🔀 Contingency Branches ('What-If' Scenarios):")
            for b in decision.contingency_branches:
                print(f"      • [{b.branch_name}]: {b.recommended_action} (Delta: {b.expected_race_time_delta:+.1f}s)")

    sim = RaceSimulator(race_df=race_df, target_driver="VER", model=model)
    results_df = sim.run_replay(on_lap_callback=print_lap_event)

    print("\n" + "=" * 70)
    print("✅ RACE REPLAY FINISHED SUCCESSFULLY!")
    print(f"Total strategy records logged: {len(results_df)}")
    print("Key strategy pivots executed:")
    pivots = results_df[results_df["PrimaryAction"].str.contains("BOX", na=False)]
    for _, row in pivots.iterrows():
        print(f"   • Lap {row['LapNumber']:02d}: {row['PrimaryAction']} -> {row['ActivePlan']} (Tyre: {row['Compound']} {row['TyreLife']} laps)")
    print("=" * 70)


if __name__ == "__main__":
    main()
