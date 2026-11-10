"""
Phase I interactive black-box dashboard (Pair A, AI-aided).

Runs the pre-trained tyre model on every curated lap, adds a set of
"what-if" probes (flip the driver flag, switch the safety-car flag on,
sweep track temperature, sweep tyre age), and writes one standalone HTML
dashboard with linked filters for the four agreed research questions.

Only inputs, outputs, labels and errors are used: no model internals.

Usage (from the repo root):
    pip install -r requirements.txt plotly   # plotly only to inline plotly.js
    python notebooks/pair_a_ai_aided/interactive_dashboard.py
Output:
    visualizations/phase1_interactive_dashboard.html  (open in any browser)
"""

from __future__ import annotations

import json
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import joblib
import numpy as np
import pandas as pd

DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "f1_curated_telemetry.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "src", "models", "trained_tyre_model.joblib")
TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "dashboard_template.html")
OUT_PATH = os.path.join(ROOT_DIR, "visualizations", "phase1_interactive_dashboard.html")

COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
TEMPS = list(range(22, 46, 2))          # covers the 21.9-44.4 °C range in the data
FUEL_STARTS = [100, 80, 60, 40, 20]     # kg at the start of the stint
MAX_LIFE = 40                           # laps simulated per stint
START_AGES = list(range(0, 31))         # tyre age when the cliff check is run
FUEL_BURN = 1.8                         # kg/lap, same constant as detect_tyre_cliff


def predict(model, frame: pd.DataFrame) -> np.ndarray:
    """Vectorised version of TyreDegradationModel.predict_delta (same clip at 0)."""
    return np.clip(model.model.predict(model._prepare_features(frame)), 0.0, None)


def lap_type(row) -> str:
    if row.IsSafetyCar:
        return "SC"
    if row.IsVSC:
        return "VSC"
    if row.IsInLap:
        return "In-lap"
    if row.IsOutLap:
        return "Out-lap"
    if row.IsYellow:
        return "Yellow"
    return "Green"


def build_lap_table(model, df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["Compound"].isin(COMPOUNDS) & df["LapTimeSeconds"].notna()].copy()
    d["LapType"] = d.apply(lap_type, axis=1)
    d["Pred"] = predict(model, d)
    d["Residual"] = d["DegradationDelta"] - d["Pred"]

    # Probe 1: same lap, driver flag flipped (VER <-> everyone else)
    flipped = d.copy()
    flipped["IsVerstappen"] = np.where(flipped["Driver"] == "VER", 0, 1)
    flipped["Driver"] = np.where(d["Driver"] == "VER", "PER", "VER")
    d["PredFlip"] = predict(model, flipped)

    # Probe 2: same lap, safety car / VSC flag switched on
    sc = d.copy()
    sc["IsSafetyCar"], sc["IsVSC"] = True, False
    d["PredSC"] = predict(model, sc)
    vsc = d.copy()
    vsc["IsSafetyCar"], vsc["IsVSC"] = False, True
    d["PredVSC"] = predict(model, vsc)

    d["RaceKey"] = d["Year"].astype(str) + " " + d["Race"]
    return d


def build_curves(model) -> dict:
    """Predicted degradation over a fresh stint for every compound/temp/fuel/driver."""
    rows = []
    for comp in COMPOUNDS:
        for t in TEMPS:
            for f in FUEL_STARTS:
                for drv in ["VER", "PER"]:
                    for k in range(1, MAX_LIFE + 1):   # tyre life 1..40, green-flag laps
                        rows.append(dict(Compound=comp, TyreLife=k, EffectiveTyreLife=k,
                                         TrackTemp=t, FuelRemainingKg=max(2.0, f - FUEL_BURN * (k - 1)),
                                         Driver=drv, IsSafetyCar=False, IsVSC=False,
                                         IsOutLap=False, _t=t, _f=f))
    g = pd.DataFrame(rows)
    g["Pred"] = predict(model, g)
    out = {}
    for (comp, t, f, drv), grp in g.groupby(["Compound", "_t", "_f", "Driver"]):
        out[f"{comp}|{t}|{f}|{drv}"] = grp["Pred"].round(3).tolist()
    return out


def build_cliffs(model) -> dict:
    """
    Re-implements model.detect_tyre_cliff in vectorised form (identical logic:
    tyre_life = effective_life = start age, fuel -1.8 kg/lap, threshold 0.22 s,
    driver left at the default 'VER') and returns the predicted cliff lap for
    each start age. Cross-checked against the original method below.
    """
    rows = []
    for comp in COMPOUNDS:
        for t in TEMPS:
            for f in FUEL_STARTS:
                for a in START_AGES:
                    for k in range(0, 40):
                        rows.append(dict(Compound=comp, TyreLife=a + k, EffectiveTyreLife=a + k,
                                         TrackTemp=t, FuelRemainingKg=max(2.0, f - FUEL_BURN * k) if k else f,
                                         Driver="VER", _t=t, _f=f, _a=a, _k=k))
    g = pd.DataFrame(rows)
    g["Pred"] = predict(model, g)
    out = {}
    for (comp, t, f), grp in g.groupby(["Compound", "_t", "_f"]):
        laps, pens = [], []
        for a, s in grp.groupby("_a"):
            p = s.sort_values("_k")["Pred"].to_numpy()
            jumps = np.where(np.diff(p) >= 0.22)[0]
            if len(jumps):
                laps.append(int(jumps[0] + 1))
                pens.append(round(float(p[jumps[0] + 1]), 3))
            else:
                laps.append(35)
                pens.append(round(float(p[-1]), 3))
        out[f"{comp}|{t}|{f}"] = {"laps": laps, "penalty": pens}

    # sanity check against the repo's own implementation
    for comp, t, f, a in [("SOFT", 32, 100, 0), ("MEDIUM", 40, 60, 10), ("HARD", 24, 80, 20)]:
        ref = model.detect_tyre_cliff(comp, a, t, f)[0]
        mine = out[f"{comp}|{t}|{f}"]["laps"][a]
        assert ref == mine, f"cliff mismatch {comp} {t} {f} {a}: {ref} vs {mine}"
    return out


def build_temp_pdp(model, laps: pd.DataFrame) -> dict:
    """Average prediction when every green lap is replayed at a fixed track temperature."""
    green = laps[laps["LapType"] == "Green"]
    out = {}
    for comp in COMPOUNDS:
        base = green[green["Compound"] == comp]
        vals = []
        for t in TEMPS:
            x = base.copy()
            x["TrackTemp"] = t
            vals.append(round(float(predict(model, x).mean()), 3))
        out[comp] = vals
    return out


def main():
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    print(f"Loaded {len(df):,} laps; model = {type(model.model).__name__}")

    laps = build_lap_table(model, df)
    print("Lap table and probes done")
    curves = build_curves(model)
    print("Stint curves done")
    cliffs = build_cliffs(model)
    print("Cliff sweep done (matches detect_tyre_cliff)")
    pdp = build_temp_pdp(model, laps)

    cols = ["RaceKey", "Driver", "LapNumber", "Compound", "TyreLife", "EffectiveTyreLife",
            "TrackTemp", "FuelRemainingKg", "LapType", "LapTimeSeconds", "DegradationDelta",
            "Pred", "Residual", "PredFlip", "PredSC", "PredVSC"]
    lap_records = laps[cols].round(3).to_dict(orient="split")["data"]

    payload = {
        "meta": {
            "modelClass": type(model.model).__name__,
            "nLaps": int(len(laps)),
            "races": sorted(laps["RaceKey"].unique().tolist()),
            "temps": TEMPS, "fuels": FUEL_STARTS, "maxLife": MAX_LIFE,
            "startAges": START_AGES,
        },
        "cols": cols,
        "laps": lap_records,
        "curves": curves,
        "cliffs": cliffs,
        "pdp": pdp,
    }

    with open(TEMPLATE_PATH, encoding="utf-8") as fh:
        html = fh.read()
    html = html.replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":")))

    # Inline plotly.js when the Python plotly package is installed, so the
    # dashboard also works offline (e.g. while recording the video).
    # Pass --cdn to keep the small CDN-loaded version instead.
    cdn_tag = '<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>'
    if "--cdn" not in sys.argv:
        try:
            import plotly
            js_path = os.path.join(os.path.dirname(plotly.__file__), "package_data", "plotly.min.js")
            with open(js_path, encoding="utf-8") as fh:
                html = html.replace(cdn_tag, "<script>" + fh.read() + "</script>")
            print("plotly.js inlined (offline-ready)")
        except ImportError:
            print("plotly not installed: dashboard loads plotly.js from the CDN")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"Dashboard written to {OUT_PATH} ({os.path.getsize(OUT_PATH) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
