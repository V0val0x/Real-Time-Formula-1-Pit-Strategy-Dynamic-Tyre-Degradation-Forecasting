# Formula 1 Tyre Degradation & Real-Time Strategy System
### AI and Visualization Lab Project (JKU Linz - Fall 2026)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastF1](https://img.shields.io/badge/telemetry-FastF1-red.svg)](https://github.com/theOehrly/Fast-F1)
[![Course](https://img.shields.io/badge/course-AI%20%26%20Vis-brightgreen.svg)](https://jku-vds-lab.at/)

An end-to-end telemetry pipeline, physics-informed machine learning model, and real-time contingency strategy engine for analyzing **tyre degradation and pit stop dynamics in Formula 1**, focused on **Red Bull Racing (Max Verstappen vs Sergio Perez)** across the 2023–2024 seasons.

---

## Repository Structure

```
AIandVis/
├── data/
│   └── processed/
│       └── f1_curated_telemetry.csv  # 9,202 curated FastF1 laps across 7 Grand Prix
├── src/
│   ├── data/
│   │   └── f1_data.py               # FastF1 loader and feature engineering pipeline
│   ├── models/
│   │   ├── tyre_model.py            # Physics-aware degradation model with cliff detector
│   │   └── trained_tyre_model.joblib # Serialized model artifact (R² = 0.724, MAE = 0.68s)
│   ├── strategy/
│   │   ├── strategy_engine.py       # Real-time multi-branch strategy recommender
│   │   └── pit_dynamics.py          # Pit-lane loss and cold tyre warm-up models
│   └── simulation/
│       └── race_simulator.py        # Lap-by-lap telemetry replay and strategy simulator
├── notebooks/
│   ├── phase1_visual_analytics.py   # Telemetry vs model behavior diagnostics
│   └── red_bull_analysis.py         # Teammate comparison (VER vs PER)
├── ai_logs/
│   └── phase1_ai_log_template.md    # Required prompt and reflection log for AI-aided pair
├── visualizations/                  # Diagnostic plots and figures
├── scripts/
│   ├── download_real_data.py        # Batch telemetry download script
│   └── train_model.py               # Model training and validation script
├── test_pipeline.py                 # End-to-end verification and demonstration script
├── requirements.txt                 # Python dependencies
├── PHASE1_GUIDE.md                  # Comprehensive Phase I instructions
└── README.md
```

---

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/V0val0x/Real-Time-Formula-1-Pit-Strategy-Dynamic-Tyre-Degradation-Forecasting.git
cd Real-Time-Formula-1-Pit-Strategy-Dynamic-Tyre-Degradation-Forecasting
```

### 2. Set Up Virtual Environment
```bash
# Create environment
python3 -m venv .venv

# Activate environment
source .venv/bin/activate  # macOS / Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

> **Note:** The preprocessed dataset (`data/processed/f1_curated_telemetry.csv`) and the trained model (`src/models/trained_tyre_model.joblib`) are included in the repository. Both pairs start with identical data and model artifacts without needing to run initial downloads.

---

## Execution

### Real-Time Race Simulation
Replays the 2023 Bahrain GP for Max Verstappen, dynamically evaluating pit windows, Safety Car opportunities, and competitor undercut threats:
```bash
python test_pipeline.py
```

### Red Bull Teammate Comparison (VER vs PER)
Generates degradation profiles and residual distributions comparing Verstappen and Perez:
```bash
python notebooks/red_bull_analysis.py
```

### Phase I Behavioral Diagnostics
Generates multi-panel telemetry vs prediction curves and strategy pivot timelines:
```bash
python notebooks/phase1_visual_analytics.py
```

---

## Phase I Investigation Guide

Detailed research questions, pair division protocols (AI-Aided vs Human-Only), and midpoint video deliverables are documented in:  
👉 **[PHASE1_GUIDE.md](PHASE1_GUIDE.md)**

---

## Key Deadlines

* **21.10.2026 (23:59):** Group formation registration.
* **28.10.2026 (23:59):** Task & model selection submission.
* **11.11.2026 (23:59):** **Midpoint Submission (Phase I)** — code, AI logs, 5-min update video.
* **09.12.2026 (23:59):** **Final Submission (Phase II)** — full code, all logs, 10-min presentation.
