# 🏏 IPL Analytics Hub — 18 Seasons of Insights
IPL Analytics is a data-driven project designed to analyze and visualize performance insights from the Indian Premier League (IPL).

---

## 📌 Project Overview

This project explores the Indian Premier League (IPL) through deep data analysis and an interactive Streamlit dashboard. It combines two raw datasets — ball-by-ball deliveries and match-level records — into a unified dataset, then surfaces insights through six interactive dashboard tabs covering batting, bowling, game outcomes, momentum shifts, and team profiles.

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
├── Data_Preparation.ipynb                     # Data cleaning, EDA, and dataset merging
├── Dashboard_Application.py                   # Streamlit dashboard application
├── requirements.txt                           # Python dependencies
└── README.md
```

---

## 📊 Dashboard Tabs

| Tab | Description |
|-----|-------------|
| **IPL Overview** | Season-wise match counts, average runs, top cities, toss decisions |
| **Batting Stats** | Top batsmen, strike rates, phase-weighted impact scores, clutch performers |
| **Bowling Performance** | Wicket leaders, economy rates, dismissal type breakdowns |
| **Game Outcome** | Toss vs. match win analysis, outcome types, venue impact |
| **Game Changers** | Momentum swing detector, collapse windows, momentum intensity curve |
| **Teams Hub** | Per-team profiles: seasons played, wins, toss wins, total runs & wickets |

### 🔘 Sidebar Filters
- **Season Range Slider** — Filter data from any range within 2007–2025
- **Team Selector** — Drill down into a specific franchise across all tabs

---

## 🧹 Data Preparation (`Data_Preparation.ipynb`)

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

## 📈 Key Insights

- Matches per season grew from ~55 (2007) to ~75 (2025)
- Teams predominantly choose to **field first** after winning the toss
- **Mumbai** hosts the most IPL matches of any city
- More matches are won by **wickets** (chasing teams) than by runs
- Defending teams win by ~30 runs on average; chasing teams win by ~6 wickets
- Most dismissals are **caught** out
- Toss win translates to match win roughly **50%** of the time — limited advantage

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3 | Core language |
| Pandas | Data manipulation |
| Matplotlib / Seaborn | Static visualisations (EDA) |
| Plotly / Plotly Express | Interactive charts in dashboard |
| Streamlit | Dashboard application framework |
| Jupyter Notebook | Data preparation and EDA |

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

### 3. Prepare the dataset
Run `Data_Preparation.ipynb` end-to-end to generate `dataset/IPL_dataset.csv`.  
*(Skip this step if `IPL_dataset.csv` is already present in the `dataset/` folder.)*

### 4. Launch the dashboard
```bash
streamlit run Dashboard_Application.py
```

---

## 📦 Requirements

```
streamlit
pandas
matplotlib
seaborn
plotly
numpy
jupyter
```

---

## 🏆 Hackathon

This project was submitted for the **ORU IPL Analytics Hackathon**.  
Data covers all IPL seasons from **2007 through 2025** (18 seasons, ball-by-ball).

---

