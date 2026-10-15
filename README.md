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
│       └── f1_curated_telemetry.csv   # 9,202 curated FastF1 laps across 7 Grand Prix
├── src/
│   ├── data/
│   │   └── f1_data.py                # FastF1 loader and feature engineering pipeline
│   ├── models/
│   │   ├── tyre_model.py             # Physics-aware degradation model with cliff detector
│   │   └── trained_tyre_model.joblib # Serialized model artifact (R² = 0.724, MAE = 0.68s)
│   ├── strategy/
│   │   ├── strategy_engine.py        # Real-time multi-branch strategy recommender
│   │   └── pit_dynamics.py           # Pit-lane loss and cold tyre warm-up models
│   └── simulation/
│       └── race_simulator.py         # Lap-by-lap telemetry replay and strategy simulator
├── notebooks/
│   ├── pair_a_ai_aided/              # Workspace for AI-Aided Pair (Phase I)
│   │   ├── phase1_visual_analytics.py
│   │   └── red_bull_analysis.py
│   └── pair_b_human_only/            # Workspace for Human-Only Pair (Phase I)
│       └── starter_analysis.py
├── ai_logs/
│   └── phase1_ai_log_template.md     # Template for logging AI interactions (Pair A)
├── visualizations/                   # Output diagnostic plots and figures
├── scripts/
│   ├── download_real_data.py         # Batch telemetry download script
│   └── train_model.py                # Model training and validation script
├── test_pipeline.py                  # End-to-end race simulation demo
├── requirements.txt                  # Python dependencies
├── PHASE1_GUIDE.md                   # Phase I requirements and deliverables guide
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
# Create virtual environment
python3 -m venv .venv

# Activate environment
source .venv/bin/activate  # macOS / Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

> **Note:** The curated dataset (`data/processed/f1_curated_telemetry.csv`) and trained model (`src/models/trained_tyre_model.joblib`) are pre-packaged in the repository so both pairs start with identical data without manual downloads.

---

## Running the Pipeline

### Shared Simulation Demo
Runs an end-to-end race replay with live strategy evaluation:
```bash
python test_pipeline.py
```

### Pair A (AI-Aided) Workspace
```bash
# Run visual analytics scripts
python notebooks/pair_a_ai_aided/phase1_visual_analytics.py
python notebooks/pair_a_ai_aided/red_bull_analysis.py
```

### Pair B (Human-Only) Workspace
```bash
# Run starter analysis script
python notebooks/pair_b_human_only/starter_analysis.py
```

---

## Phase I Investigation Guide

For full project requirements, open research questions, pair rules, and midpoint video deliverables, refer to:  
👉 **[PHASE1_GUIDE.md](PHASE1_GUIDE.md)**

---

## Git Branching & Collaboration Workflow

To ensure smooth collaboration between the two pairs and prevent merge conflicts, all team members should follow this workflow:

### 1. Branch Naming Convention
* **Main Branch (`main`):** Production-ready, shared baseline code. No direct untested pushes to `main`.
* **Pair A Work:** `pair-a/<topic-or-task>` (e.g. `pair-a/interactive-ui`, `pair-a/phase1-telemetry-plots`).
* **Pair B Work:** `pair-b/<topic-or-task>` (e.g. `pair-b/residual-analysis`, `pair-b/phase1-static-plots`).
* **Shared Core Fixes:** `core/<bugfix-or-feature>` (e.g. `core/fix-pit-delta`, `core/data-loader-patch`).

### 2. Workspace Isolation Rule
* **Pair A** works strictly inside `notebooks/pair_a_ai_aided/` and `ai_logs/`.
* **Pair B** works strictly inside `notebooks/pair_b_human_only/`.
* Shared core files (`src/`, `data/`, `scripts/`) must only be modified with explicit alignment from all 4 team members.

### 3. Step-by-Step Git Commands

#### A. Starting New Work
```bash
# 1. Fetch latest updates from main
git checkout main
git pull origin main

# 2. Branch off for your pair's task
git checkout -b pair-a/phase1-visuals  # or pair-b/phase1-visuals
```

#### B. Committing and Pushing
```bash
# 3. Check status and stage only your files
git status
git add notebooks/pair_a_ai_aided/

# 4. Commit with a clear, concise message
git commit -m "feat(pair-a): add interactive degradation curve plot"

# 5. Push your branch to GitHub
git push -u origin pair-a/phase1-visuals
```

#### C. Merging via Pull Request (PR)
1. On GitHub, create a **Pull Request** from your branch into `main`.
2. Confirm that only files in your pair's directory are modified.
3. Have at least one team member from the other pair review and approve the PR.
4. Merge using **Squash and Merge** (or Rebase) to maintain a clean linear history.
5. After merging, update your local `main`:
   ```bash
   git checkout main
   git pull origin main
   git branch -d pair-a/phase1-visuals
   ```

---

## Key Deadlines

* **21.10.2026 (23:59):** Group formation registration.
* **28.10.2026 (23:59):** Task & model selection submission.
* **11.11.2026 (23:59):** **Midpoint Submission (Phase I)** — code, AI logs, 5-min update video.
* **09.12.2026 (23:59):** **Final Submission (Phase II)** — full code, all logs, 10-min presentation.
