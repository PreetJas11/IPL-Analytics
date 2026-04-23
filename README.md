# 🏏 IPL Analytics Hub — 18 Seasons of Insights + Match Prediction
-------------------------------------------------------------------------
IPL Analytics is a data-driven project designed to analyze, visualize and predict performance insights from the Indian Premier League (IPL). The project runs in two phases:

- **Phase 1 — Descriptive analytics:** cleaning ball-by-ball + match data and surfacing 18 seasons of insights through an interactive six-tab Streamlit dashboard.
- **Phase 2 — Predictive analytics:** three ML models on top of the same dataset that predict the **toss winner**, the **pre-match winner** (with SHAP explanations), and the **live in-match winner** over by over — delivered.

---

## 📌 Project Overview

This project explores the IPL through deep data analysis AND machine learning. Two raw datasets — ball-by-ball deliveries and match-level records — are merged into a unified dataset. Phase 1 surfaces insights through 6 interactive dashboard tabs covering batting, bowling, game outcomes, momentum shifts and team profiles. Phase 2 trains three models on the same data and adds a 7th tab that predicts the toss winner, the match winner (with SHAP explanations) and live in-game win probability.

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         RAW DATA                                           │
│   deliveries_updated_ipl_upto_2025.csv   matches_updated_ipl_upto_2025.csv │
└───────────────────────────────┬────────────────────────────────────────────┘
                                │
                                ▼
              ┌───────────────────────────────────┐
              │  PHASE 1  —  Data Preparation     │
              │   Data_Preparation.ipynb          │
              │   • null handling   • team rename │
              │   • season parsing  • dedup       │
              │   • deliveries ⋈ matches (merge)  │
              └───────────────┬───────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │ IPL_dataset.csv     │
                    │ (unified dataset)   │
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
  ┌────────────────────────┐      ┌────────────────────────────────┐
  │  PHASE 1 — Dashboard   │      │  PHASE 2 — Model Training       │
  │  Dashboard_Application │      │  train_models.py                │
  │  .py (Streamlit)       │      │  train_live_model.py            │
  │                        │      │    • feature engineering        │
  │  Tab 1  IPL Overview   │      │    • LR_L1 / LR_L2 / XGBoost    │
  │  Tab 2  Batting Stats  │      │    • log-loss model selection   │
  │  Tab 3  Bowling        │      │    • current-season roster      │
  │  Tab 4  Game Outcome   │      └────────────────┬────────────────┘
  │  Tab 5  Game Changers  │                       │
  │  Tab 6  Teams Hub      │                       ▼
  │                        │         ┌─────────────────────────────┐
  │                        │         │ models/                     │
  │                        │         │  ├── toss_winner_model.pkl  │
  │                        │         │  ├── match_winner_model.pkl │
  │                        │         │  ├── live_match_model.pkl   │
  │                        │         │  ├── encoders.pkl           │
  │                        │         │  ├── team_stats.pkl         │
  │                        │         │  └── meta.json              │
  │                        │         └────────────────┬────────────┘
  │                        │                          │
  │                        │                          ▼
  │                        │         ┌─────────────────────────────┐
  │                        │         │  predict.py  (inference)    │
  │                        │         │   • predict_toss_winner     │
  │                        │         │   • predict_match_winner    │
  │                        │         │   • explain_prediction SHAP │
  │                        │         │   • predict_live_match      │
  │                        │         └────────────────┬────────────┘
  │                        │                          │
  │                        │                          ▼
  │                        │         ┌─────────────────────────────┐
  │  Tab 7 ────────────────┼────────►│  tab7_prediction.py         │
  │  Prediction            │         │   • team/venue pickers      │
  │                        │         │   • toss + match + SHAP     │
  │                        │         │   • live (@st.fragment)     │
  └────────────────────────┘         └─────────────────────────────┘
                              │
                              ▼
                        ┌───────────┐
                        │   USER    │
                        └───────────┘
```
---

## 📁 Project Structure

```
IPL-Analytics/
├── dataset/              # raw CSVs + merged IPL_dataset.csv
├── models/               # trained .pkl artifacts (Phase 2)
├── Data_Preparation.ipynb
├── Dashboard_Application.py
├── features.py · train_models.py · train_live_model.py
├── predict.py · tab7_prediction.py
├── requirements.txt
└── README.md
```

---

## 📊 Dashboard Tabs

| # | Tab | Phase | What it shows |
|---|-----|-------|---------------|
| 1 | IPL Overview        | 1 | Season counts, avg runs, top cities, toss decisions |
| 2 | Batting Stats       | 1 | Top batsmen, strike rates, clutch performers |
| 3 | Bowling Performance | 1 | Wicket leaders, economy, dismissal breakdown |
| 4 | Game Outcome        | 1 | Toss vs. win, outcome types, venue impact |
| 5 | Game Changers       | 1 | Momentum swings, collapse windows |
| 6 | Teams Hub           | 1 | Per-team profile (seasons, wins, runs, wickets) |
| 7 | 🔮 **Prediction**   | 2 | Toss + match winner (SHAP) + live win probability |

**Sidebar:** season-range slider · team selector · current-season roster filter in Tab 7.

---

## 🧹 Phase 1 — Data Prep

- Null handling, dedup, Super Over removal, team-name standardisation (`Bangalore → Bengaluru`, `Kings XI → Punjab Kings`, `Daredevils → Capitals`)
- Derived flags: `is_wicket`, `valid_ball`, `total_runs`, `toss_win`
- Merged deliveries ⋈ matches on `match_id` → `IPL_dataset.csv`

## 🤖 Phase 2 — Models

| Task | Model | Accuracy | Log-loss |
|------|-------|---------:|---------:|
| Toss winner      | LR (L1) | ~54 % | 0.688 |
| Pre-match winner | LR (L1) | ~63 % | 0.638 |
| **Live winner**  | **XGBoost** | **70.0 %** | **0.546** |

- **Features:** team strength, H2H, form, venue win rate, toss, momentum, difference features to reduce collinearity.
- **Live model:** 46,380 over-by-over snapshots, state-only features (score, wickets, overs, RR, required-RR).
- **Explainability:** SHAP bar chart + plain-English summary in Tab 7.
- **UX:** latched Predict button + `@st.fragment` so live prediction re-runs only its own block.

---

## 🚀 Getting Started

```bash
# 1. install
pip install -r requirements.txt

# 2. (optional) regenerate merged dataset
jupyter nbconvert --execute Data_Preparation.ipynb

# 3. (optional) retrain models
python train_models.py
python train_live_model.py

# 4. run the dashboard
streamlit run Dashboard_Application.py
```

## 🛠️ Tech Stack

Python · Pandas · NumPy · Streamlit · Plotly · Altair · scikit-learn · XGBoost · SHAP · Jupyter

---

## 🏆 Hackathon

Submitted for the **ORU IPL Analytics Hackathon** · Data: all IPL seasons 2007–2025.

## 👩‍💻 Author

**Jaspreet Kaur** 
