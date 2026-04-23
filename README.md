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
│
├── dataset/
│   ├── deliveries_updated_ipl_upto_2025.csv   # Ball-by-ball delivery data
│   ├── matches_updated_ipl_upto_2025.csv      # Match-level records
│   └── IPL_dataset.csv                        # Final merged dataset (output of notebook)
│
├── models/                                    # Phase 2 — trained artifacts
│   ├── toss_winner_model.pkl
│   ├── match_winner_model.pkl
│   ├── live_match_model.pkl
│   ├── encoders.pkl
│   ├── team_stats.pkl
│   └── meta.json
│
├── Data_Preparation.ipynb                     # Phase 1 — cleaning, EDA, dataset merging
├── Dashboard_Application.py                   # Streamlit dashboard (all 7 tabs)
│
├── features.py                                # Phase 2 — feature engineering
├── train_models.py                            # Phase 2 — pre-match + toss models
├── train_live_model.py                        # Phase 2 — live (over-by-over) model
├── predict.py                                 # Phase 2 — inference + SHAP
├── tab7_prediction.py                         # Phase 2 — prediction tab UI
│
├── requirements.txt                           # Python dependencies
└── README.md
```

---

## 📊 Dashboard Tabs

| Tab | Phase | Description |
|-----|-------|-------------|
| **IPL Overview**       | 1 | Season-wise match counts, average runs, top cities, toss decisions |
| **Batting Stats**      | 1 | Top batsmen, strike rates, phase-weighted impact scores, clutch performers |
| **Bowling Performance**| 1 | Wicket leaders, economy rates, dismissal type breakdowns |
| **Game Outcome**       | 1 | Toss vs. match win analysis, outcome types, venue impact |
| **Game Changers**      | 1 | Momentum swing detector, collapse windows, momentum intensity curve |
| **Teams Hub**          | 1 | Per-team profiles: seasons played, wins, toss wins, total runs & wickets |
| **🔮 Prediction**      | 2 | Toss + pre-match winner (with SHAP) + live over-by-over win probability |

### 🔘 Sidebar Filters
- **Season Range Slider** — Filter data from any range within 2007–2025
- **Team Selector** — Drill down into a specific franchise across all tabs
- **Current-season only** — The Prediction tab restricts Team A / Team B pickers to the 10 IPL 2025 franchises (defunct franchises are hidden)

---

## 🧹 Phase 1 — Data Preparation (`Data_Preparation.ipynb`)

### Deliveries Dataset
- Imputed null values in `isWide`, `isNoBall`, `Byes`, `LegByes`, `Penalty` with `0`
- Parsed `date` column and extracted `season` (year)
- Removed duplicate deliveries (by `matchId`, `inning`, `over`, `ball`)
- Removed Super Over deliveries (`inning > 2`)
- Filled missing `dismissal_kind` with `"not out"`; derived `is_wicket` flag
- Computed `total_runs` = `batsman_runs + extras`
- Computed `valid_ball` flag (excludes wides and no-balls)
- Standardised team names for consistency across seasons:
  - `Kings XI Punjab` → `Punjab Kings`
  - `Delhi Daredevils` → `Delhi Capitals`
  - `Royal Challengers Bangalore` → `Royal Challengers Bengaluru`

### Matches Dataset
- Imputed missing `city` values using venue-to-city mapping (Dubai, Sharjah)
- Filled `neutralvenue`, `eliminator`, `method`, and `outcome` nulls with appropriate defaults
- Extracted clean 4-digit `season` from mixed-format season strings
- Derived `toss_win` boolean column
- Dropped redundant `date1`, `date2` columns
- Applied same team name standardisation as deliveries

### Dataset Merge
- Merged deliveries and matches on `match_id` (left join)
- Resolved duplicate `date` and `season` columns from both sources
- Final dataset saved as `dataset/IPL_dataset.csv`

---

## 🤖 Phase 2 — Predictive Analytics

Phase 2 adds three models on top of the unified dataset. All three share the same data-prep output; no new raw data is required.

### 2.1 Feature Engineering (`features.py`)

Ball-by-ball rows are aggregated up to **team-match** rows, then enriched with **time-aware** signals so the model never peeks into the future:

| Feature family | Examples |
|---|---|
| Team strength   | H2H win rate, recent-10-match form, batting / bowling ratings |
| Venue           | Team-at-venue win rate, average 1st-innings score, chase success |
| Matchup         | Strength difference, form difference, streak difference |
| Toss            | Toss winner, toss decision (bat / field) |
| Season          | Team's season-so-far record, momentum score |

Difference features (`strength_diff`, `form_diff`, `streak_diff`, …) replace raw pairs to reduce collinearity and make SHAP attributions cleaner.

### 2.2 Pre-match Models (`train_models.py`)

Two classifiers trained on **1,146 matches** (chronological train/test split so future info never leaks):

| Task | Best model | Accuracy | Log-loss |
|------|-----------|----------|----------|
| **Toss winner**  | Logistic Regression (L1) | ~54 % | 0.688 |
| **Match winner** | Logistic Regression (L1) | ~63 % | 0.638 |

L1 was selected over L2 and XGBoost on held-out **log-loss** — it produced better-calibrated probabilities despite similar top-1 accuracy.

### 2.3 Live-match Model (`train_live_model.py`)

Walks the ball-by-ball CSV and snapshots match state at the end of every over in both innings, producing **46,380 training rows** across all matches. Features are **state-only** (no pre-match team strength) because in-game state dominates once balls have been bowled:

```
inning, overs_done, balls_remaining, score, wickets_in_hand,
current_rr, is_inn2, target, runs_needed, req_rr, rr_diff,
wickets_per_over
```

A **match-ID-aware train/test split** prevents snapshot leakage across splits (all snapshots from a given match land on the same side).

| Model | Accuracy | Log-loss |
|---|---|---|
| LR (L2) | 66.4 % | 0.603 |
| LR (L1) | 66.1 % | 0.608 |
| **XGBoost** ✅ | **70.0 %** | **0.546** |

### 2.4 Inference + SHAP (`predict.py`)

| Function | What it returns |
|---|---|
| `predict_toss_winner(team_a, team_b, venue)` | `{prob_a, prob_b}` |
| `predict_match_winner(team_a, team_b, venue, toss_winner?, toss_decision?)` | `{prob_a, prob_b, toss_winner_used}` — marginalises over toss if unknown |
| `explain_prediction(...)` | `{prediction, contributions[], summary}` — SHAP values + plain-English summary |
| `predict_live_match(team_a, team_b, venue, current_score, wickets, overs, batting_team, target?, inning?)` | `{prob_a, prob_b, commentary, state}` |
| `list_active_teams()` / `current_season()` | Restrict UI to the 10 IPL 2025 franchises |

SHAP contributions are rendered as a horizontal bar chart colored by which team each feature favors, plus an auto-generated plain-English summary grouping factors by "pros / cons" for Team A.

### 2.5 Prediction Tab UI (`tab7_prediction.py`)

- **Current-season roster only** — Team A / Team B pickers hide defunct franchises.
- **Latched Predict button** — once clicked, the result persists in `st.session_state` so adjusting inputs later re-renders without re-clicking.
- **Live block is an `@st.fragment`** — clicking "Predict live" only re-runs the bottom live block; the toss / match / SHAP section above never reloads.

---

## 📈 Key Insights (Phase 1)

- Matches per season grew from ~55 (2007) to ~75 (2025)
- Teams predominantly choose to **field first** after winning the toss
- **Mumbai** hosts the most IPL matches of any city
- More matches are won by **wickets** (chasing teams) than by runs
- Defending teams win by ~30 runs on average; chasing teams win by ~6 wickets
- Most dismissals are **caught** out
- Toss win translates to match win roughly **50 %** of the time — limited advantage (and our toss-as-a-feature model confirms this: toss contributes little to match-winner SHAP)

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3             | Core language |
| Pandas / NumPy       | Data manipulation |
| Matplotlib / Seaborn | Static visualisations (EDA) |
| Plotly / Plotly Express | Interactive charts in dashboard |
| Altair               | SHAP bar chart in Prediction tab |
| Streamlit            | Dashboard application framework |
| Jupyter Notebook     | Data preparation and EDA |
| scikit-learn         | Logistic Regression, train/test split, metrics |
| XGBoost              | Live-match gradient-boosted classifier |
| SHAP                 | Model explanations |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/ipl-analytics.git
cd ipl-analytics
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset (Phase 1)
Run `Data_Preparation.ipynb` end-to-end to generate `dataset/IPL_dataset.csv`.
*(Skip this step if `IPL_dataset.csv` is already present in the `dataset/` folder.)*

### 4. Train the models (Phase 2)
```bash
python train_models.py        # toss + pre-match winner models
python train_live_model.py    # live (over-by-over) model
```
Artifacts are written to `models/`. Skip this step if `models/` already contains the `.pkl` files.

### 5. Launch the dashboard
```bash
streamlit run Dashboard_Application.py
```
The dashboard opens at http://localhost:8501 with all 7 tabs.

---

## 📦 Requirements

```
streamlit
pandas
numpy
matplotlib
seaborn
plotly
altair
scikit-learn
xgboost
shap
jupyter
```

---

## 🏆 Hackathon

This project was submitted for the **ORU IPL Analytics Hackathon**.
Data covers all IPL seasons from **2008 through 2025** (18 seasons, ball-by-ball).

Phase 1 delivers a full descriptive-analytics dashboard across 6 tabs.
Phase 2 adds predictive analytics (toss, pre-match, live) with SHAP explanations in a 7th tab.

👩‍💻 Author
Built by Jaspreet Kaur
