# Phase I Visual Analytics Guide (Vis4AI & AI4Vis)
> **Course:** AI and Visualization (JKU Linz)  
> **Topic:** Real-Time Formula 1 Pit Strategy & Dynamic Tyre Degradation Forecasting (Red Bull Racing Case Study)  
> **Midpoint Deadline (Phase I Submission):** 11.11.2026 (23:59)

---

## 1. Goal and Core Question
**Phase I Question:** *How does the model behave?*

The model is investigated as a **black box**: internal weights and representations are ignored for now, focusing strictly on inputs, outputs, prediction residuals, and strategic decisions.

> **Course Guideline:**  
> Visualization is an investigative tool, not a benchmark summary. A simple table of metrics (e.g. MAE or R²) is insufficient. The visualizations must expose systematic behavioral patterns, error clusters, and domain-specific anomalies.

---

## 2. Research Questions & Hypotheses

1. **Driver Bias (VER vs PER):**
   * How does the model perform on identical Red Bull machinery (RB19/RB20) across different driving styles?
   * *Observation:* The model overestimates degradation for Verstappen in clean air (mean residual +0.34s) while showing higher variance for Perez fighting in traffic.
2. **Disturbance Robustness (Safety Car & VSC):**
   * Does the model differentiate nominal tyre age from effective wear during caution periods?
3. **Non-Linear Tyre Cliff:**
   * At which tyre life do Soft, Medium, and Hard compounds reach their performance cliff? Does the model flag cliffs before irreversible pace loss?
4. **Pit Cycle & Cold Tyre Dynamics:**
   * Where are prediction residuals concentrated: on in-lap deceleration or out-lap cold tyre warm-up? How does track temperature (< 28°C) affect Hard compound warm-up errors?

---

## 3. Pair Protocols (AI4Vis Track)

Both pairs use the same processed dataset (`data/processed/f1_curated_telemetry.csv`) and trained model (`src/models/trained_tyre_model.joblib`), but execute the visual analysis independently.

```
                      ┌──────────────────────────────────────────────┐
                      │                Shared Base:                  │
                      │  9,202 F1 Laps Telemetry + Trained ML Model  │
                      └──────────────────────┬───────────────────────┘
                                             │
                     ┌───────────────────────┴───────────────────────┐
                     ▼                                               ▼
       ┌───────────────────────────┐                   ┌───────────────────────────┐
       │     Pair A: AI-Aided      │                   │    Pair B: Human-Only     │
       ├───────────────────────────┤                   ├───────────────────────────┤
       │ • Uses Generative AI      │                   │ • No GenAI for code or    │
       │   (ChatGPT / Claude)      │                   │   visual design           │
       │ • Rich interactive UIs    │                   │ • Light code completion   │
       │   (Plotly / Streamlit)    │                   │   and docs search allowed │
       │ • Mandatory AI log in     │                   │ • Static / exploratory    │
       │   ai_logs/                │                   │   plots (Matplotlib/Seaborn)
       └───────────────────────────┘                   └───────────────────────────┘
```

### Guidelines for Pair A (AI-Aided)
* Use Generative AI for hypothesis ideation, dashboard development, and interactive controls.
* Higher interaction fidelity (interactive widgets, dynamic filters) is expected.
* **Mandatory AI Log:** Document key prompts, accepted/rejected suggestions, and failure modes in `ai_logs/phase1_ai_log.md` using the provided template.

### Guidelines for Pair B (Human-Only)
* Implement analytical code independently without GenAI code generation or automated visual design.
* Standard plotting tools (Matplotlib, Seaborn) are completely appropriate. Depth of reasoning and domain accuracy matter most.

---

## 4. Midpoint Deliverable Checklist (Deadline: 11.11.2026 23:59)

1. **Code:**
   * Pair A code / notebooks (`notebooks/pair_a_ai_aided/`).
   * Pair B code / notebooks (`notebooks/pair_b_human_only/`).
2. **AI Interaction Record:**
   * Completed `ai_logs/phase1_ai_log.md`.
3. **5-Minute Progress Update Video:**
   * *0:00–1:00:* Problem statement and Red Bull case motivation.
   * *1:00–2:30:* Human-only pair findings and domain plots.
   * *2:30–4:00:* AI-aided pair findings and interactive visualizations.
   * *4:00–4:40:* AI4Vis reflection (where AI accelerated progress, where it hallucinated, what each pair missed).
   * *4:40–5:00:* Phase II roadmap (White-box XAI, role switch).
   * *Note: Exceeding 5 minutes results in point deductions.*
