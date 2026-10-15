# Phase I Project & Analytics Guide
> **Course:** AI and Visualization (JKU Linz - Fall 2026)  
> **Topic:** Real-Time Formula 1 Pit Strategy & Dynamic Tyre Degradation Forecasting (Red Bull Racing Case Study)  
> **Midpoint Deadline (Phase I):** 11.11.2026 at 23:59

---

## 1. Project Overview & Shared Foundation

The goal of this project is to analyze the behavior of a physics-aware machine learning model and real-time decision engine that predicts Formula 1 tyre degradation and pit-stop strategies. The study focuses on Red Bull Racing (Max Verstappen vs Sergio Perez) during the 2023–2024 seasons.

Both pairs in the group share the exact same baseline assets located in the repository:
1. **Curated Telemetry Dataset:** `data/processed/f1_curated_telemetry.csv` (9,202 clean laps across 7 representative Grand Prix).
2. **Pre-Trained Degradation Model:** `src/models/trained_tyre_model.joblib` (trained on historical lap telemetry, inputs: tyre compound, tyre age, effective wear, fuel mass, track temperature, driver indicator).
3. **Pit Dynamics & Strategy Recommender:** `src/strategy/` and `src/simulation/race_simulator.py`.

---

## 2. Phase I Objective: "How Does the Model Behave?"

In Phase I, the model is treated as a **black box**. You do not look inside internal weights, layers, or representations. You inspect inputs, outputs, errors, and decision triggers.

> **Key Rule from Syllabus:**  
> A simple performance table (e.g. MAE or R² score) is **not** sufficient. Visualizations must reveal systematic behavioral patterns, error distributions, or edge-case failures across different conditions.

---

## 3. Agreed Common Direction (Open Research Questions)

Before branching into separate pair work, the team agreed to investigate the following core behavioral questions:

1. **Driver Style & Vehicle Matching (VER vs PER):**
   * Does the model perform equally well for both drivers in identical machinery, or does it show systematic bias/error patterns for one driving style over another?
2. **Compound Degradation Profiles:**
   * How do predicted pace degradation curves compare across Soft, Medium, and Hard compounds? Where does the model predict performance cliffs?
3. **Disruption & Caution Periods (Safety Car / VSC):**
   * How does the model respond to caution periods where physical tyre wear is reduced compared to normal green flag running?
4. **Pit Cycle & Cold Tyre Dynamics:**
   * How do model errors behave on in-laps (entering the pit lane) and out-laps (cold tyre warm-up)? Does track temperature influence prediction residuals?

*Note: Each pair must visually explore these questions independently to discover their own findings without pre-determined answers.*

---

## 4. Pair Workspaces & AI4Vis Protocol

The 4 group members are divided into two pairs for Phase I:

```
                            Shared Project Base
             (Dataset: f1_curated_telemetry.csv, Model: trained_tyre_model.joblib)
                                    │
               ┌────────────────────┴────────────────────┐
               ▼                                         ▼
      Pair A (AI-Aided)                         Pair B (Human-Only)
   `notebooks/pair_a_ai_aided/`              `notebooks/pair_b_human_only/`
   • Active use of GenAI (LLMs)              • No Generative AI
   • Interactive dashboards / controls       • Rely on own reasoning & code
   • Mandatory log in ai_logs/               • Traditional static/exploratory plots
```

### Pair A (AI-Aided Pair)
- **Role:** Uses Generative AI (ChatGPT, Claude, etc.) to brainstorm visualizations, generate plotting code, and build interactive exploration widgets.
- **Expectation:** Higher interaction richness (e.g., interactive sliders, dropdowns, dynamic filtering) to reflect AI acceleration.
- **Deliverable:** Keep a log of prompts, accepted/rejected suggestions, and AI errors in `ai_logs/phase1_ai_log.md` (using `ai_logs/phase1_ai_log_template.md`).

### Pair B (Human-Only Pair)
- **Role:** Conducts the analysis independently without Generative AI assistance. Rely on standard data science libraries (Matplotlib, Seaborn, Pandas) and manual reasoning.
- **Code Completion:** Standard IDE completion and reading official library documentation are permitted; code generation prompts are prohibited.
### Git Branching Rules for Pairs
- **Never push untested code directly to `main`**.
- Pair A works on `pair-a/...` branches (modifying only `notebooks/pair_a_ai_aided/` and `ai_logs/`).
- Pair B works on `pair-b/...` branches (modifying only `notebooks/pair_b_human_only/`).
- Merge changes via GitHub Pull Requests reviewed by the other pair. See detailed git commands in [README.md](README.md#git-branching--collaboration-workflow).

---

## 5. Midpoint Deliverable Checklist (Deadline: 11.11.2026 23:59)

At the midpoint deadline, each group submits:

1. **Code from Both Pairs:**
   - Pair A code in `notebooks/pair_a_ai_aided/`.
   - Pair B code in `notebooks/pair_b_human_only/`.
2. **AI Interaction Record:**
   - Completed `ai_logs/phase1_ai_log.md` from the AI-aided pair.
3. **Recorded Progress-Update Video (Max 5 Minutes):**
   - *Minute 0:00–1:00:* Introduction to the problem, task, and dataset.
   - *Minute 1:00–2:30:* Human-only pair findings and diagnostic charts.
   - *Minute 2:30–4:00:* AI-aided pair findings and interactive visualizations.
   - *Minute 4:00–4:40:* AI4Vis reflection (comparison of processes, AI acceleration vs human domain depth, observed AI failure patterns).
   - *Minute 4:40–5:00:* Phase II preview (White-box analysis plan and pair role swap).
   - *Warning: Exceeding 5 minutes results in a point deduction.*
