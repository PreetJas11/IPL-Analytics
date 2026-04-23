
"""
IPL Match & Toss Winner Prediction — Training Script

"""
# import libraries
import json
import pickle
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from features import FeatureEncoder  # shared so pickle can find it

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[warn] xgboost not installed — comparing only LR and RF")


# CONFIG DATA
HERE = Path(__file__).parent
DATA_PATH = HERE / "dataset/IPLDataset.csv"
MODEL_DIR = HERE / "models"
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
ROLLING_WINDOW = 20     # matches used for rolling bat/bowl rates & recent form
EWMA_ALPHA     = 0.3    # weight on latest match for EWMA form
PP_LAST_OVER   = 5      # powerplay = overs 0..5 inclusive
DEATH_FIRST_OVER = 15   # death overs = overs 15..19 inclusive


# LOAD & AGGREGATE DATA
def load_matches_and_team_stats(csv_path: Path):
    df = pd.read_csv(csv_path, low_memory=False)

    # --- match-level (winner, toss, venue, date) ---
    matches = (
        df.groupby("match_id")
        .agg(
            team1=("team1", "first"),
            team2=("team2", "first"),
            toss_winner=("toss_winner", "first"),
            toss_decision=("toss_decision", "first"),
            winner=("winner", "first"),
            venue=("venue", "first"),
            season=("season", "first"),
            date=("date", "first"),
        )
        .reset_index()
    )
    matches = matches.dropna(
        subset=["winner", "toss_winner", "team1", "team2", "venue", "date"]
    )
    matches = matches[matches.apply(
        lambda r: r["winner"] in (r["team1"], r["team2"]), axis=1
    )]
    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    matches = matches.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # --- ball-by-ball batting stats per (match, team) ---
    df["is_pp"]    = df["over"] <= PP_LAST_OVER
    df["is_death"] = df["over"] >= DEATH_FIRST_OVER
    df["is_valid"] = df["valid_ball"].astype(int)

    # Masked helper columns so a single groupby-sum computes pp / death splits
    df["pp_runs"]    = df["total_runs"] * df["is_pp"]
    df["pp_balls"]   = df["is_valid"]  * df["is_pp"]
    df["death_runs"] = df["total_runs"] * df["is_death"]
    df["death_balls"]= df["is_valid"]  * df["is_death"]

    bat = (
        df.groupby(["match_id", "batting_team"], as_index=False)
        .agg(
            bat_runs       =("total_runs", "sum"),
            bat_balls      =("is_valid",   "sum"),
            bat_pp_runs    =("pp_runs",    "sum"),
            bat_pp_balls   =("pp_balls",   "sum"),
            bat_death_runs =("death_runs", "sum"),
            bat_death_balls=("death_balls","sum"),
        )
        .rename(columns={"batting_team": "team"})
    )

    # Bowling = Grouped by the bowling_team side.
    bowl = (
        df.groupby(["match_id", "bowling_team"], as_index=False)
        .agg(
            bowl_runs       =("total_runs", "sum"),
            bowl_balls      =("is_valid",   "sum"),
            bowl_pp_runs    =("pp_runs",    "sum"),
            bowl_pp_balls   =("pp_balls",   "sum"),
            bowl_death_runs =("death_runs", "sum"),
            bowl_death_balls=("death_balls","sum"),
        )
        .rename(columns={"bowling_team": "team"})
    )

    team_match_stats = bat.merge(bowl, on=["match_id", "team"], how="outer").fillna(0)
    return matches, team_match_stats


# TIME-AWARE FEATURE COMPUTATION 
def safe_div(n, d):
    return (n / d) if d else 0.0


def compute_historical_features(matches: pd.DataFrame, tms: pd.DataFrame):
    """Chronological walk. For each match, compute features using only the
    history up to that match. 
    """
    # indexable lookup: (match_id, team) -> stats row 
    tms_idx = tms.set_index(["match_id", "team"])
    stat_cols = [
        "bat_runs", "bat_balls", "bat_pp_runs", "bat_pp_balls",
        "bat_death_runs", "bat_death_balls",
        "bowl_runs", "bowl_balls", "bowl_pp_runs", "bowl_pp_balls",
        "bowl_death_runs", "bowl_death_balls",
    ]

    # per-team rolling deques (last ROLLING_WINDOW matches) 
    rolling = defaultdict(lambda: {c: deque(maxlen=ROLLING_WINDOW) for c in stat_cols})

    # per-team scalar stats 
    h2h = defaultdict(lambda: {"played": 0, "a_won": 0})
    venue_stats = defaultdict(lambda: {"played": 0, "won": 0})
    overall_stats = defaultdict(lambda: {"played": 0, "won": 0})
    streak = defaultdict(int)            # +N winning streak, -N losing streak
    ewma_form = defaultdict(lambda: 0.5) # exponentially-weighted recent winrate
    toss_stats = defaultdict(lambda: {"played": 0, "won": 0})
    toss_venue_stats = defaultdict(lambda: {"played": 0, "won": 0})

    def wr(d):
        return (d["won"] / d["played"]) if d["played"] else 0.5

    def h2h_wr_for(a, b):
        key = tuple(sorted([a, b]))
        d = h2h[key]
        if d["played"] == 0:
            return 0.5
        first = key[0]
        a_wins = d["a_won"] if first == a else d["played"] - d["a_won"]
        return a_wins / d["played"]

    def rolling_rate(team, runs_col, balls_col):
        r = sum(rolling[team][runs_col])
        b = sum(rolling[team][balls_col])
        # Convert balls to overs (6 legal balls per over) — rpo = runs / overs
        return safe_div(r * 6.0, b) if b else 0.0

    feats = []
    for _, r in matches.iterrows():
        a, b, v = r["team1"], r["team2"], r["venue"]

        feats.append({
            "h2h_winrate_a":         h2h_wr_for(a, b),
            "overall_winrate_a":     wr(overall_stats[a]),
            "overall_winrate_b":     wr(overall_stats[b]),
            "ewma_form_a":           ewma_form[a],
            "ewma_form_b":           ewma_form[b],
            "streak_a":              streak[a],
            "streak_b":              streak[b],
            "bat_rpo_a":             rolling_rate(a, "bat_runs", "bat_balls"),
            "bat_rpo_b":             rolling_rate(b, "bat_runs", "bat_balls"),
            "bowl_rpo_a":            rolling_rate(a, "bowl_runs", "bowl_balls"),
            "bowl_rpo_b":            rolling_rate(b, "bowl_runs", "bowl_balls"),
            "bat_pp_rpo_a":          rolling_rate(a, "bat_pp_runs", "bat_pp_balls"),
            "bat_pp_rpo_b":          rolling_rate(b, "bat_pp_runs", "bat_pp_balls"),
            "bowl_pp_rpo_a":         rolling_rate(a, "bowl_pp_runs", "bowl_pp_balls"),
            "bowl_pp_rpo_b":         rolling_rate(b, "bowl_pp_runs", "bowl_pp_balls"),
            "bat_death_rpo_a":       rolling_rate(a, "bat_death_runs", "bat_death_balls"),
            "bat_death_rpo_b":       rolling_rate(b, "bat_death_runs", "bat_death_balls"),
            "bowl_death_rpo_a":      rolling_rate(a, "bowl_death_runs", "bowl_death_balls"),
            "bowl_death_rpo_b":      rolling_rate(b, "bowl_death_runs", "bowl_death_balls"),
            # venue + toss historicals (used by predict & by toss model)
            "venue_winrate_a":       wr(venue_stats[(a, v)]),
            "venue_winrate_b":       wr(venue_stats[(b, v)]),
            "toss_rate_a":           wr(toss_stats[a]),
            "toss_rate_b":           wr(toss_stats[b]),
            "toss_rate_a_at_venue":  wr(toss_venue_stats[(a, v)]),
            "toss_rate_b_at_venue":  wr(toss_venue_stats[(b, v)]),
        })

        # AFTER recording features, update all stats with this match's data
        a_won = int(r["winner"] == a)
        b_won = int(r["winner"] == b)
        a_won_toss = int(r["toss_winner"] == a)
        b_won_toss = int(r["toss_winner"] == b)

        key = tuple(sorted([a, b]))
        h2h[key]["played"] += 1
        h2h[key]["a_won"] += a_won if key[0] == a else b_won

        venue_stats[(a, v)]["played"] += 1; venue_stats[(a, v)]["won"] += a_won
        venue_stats[(b, v)]["played"] += 1; venue_stats[(b, v)]["won"] += b_won

        overall_stats[a]["played"] += 1; overall_stats[a]["won"] += a_won
        overall_stats[b]["played"] += 1; overall_stats[b]["won"] += b_won

        # streak: +N = N wins in a row, -N = N losses in a row
        def _update_streak(s, won):
            if won:
                return s + 1 if s > 0 else 1
            else:
                return s - 1 if s < 0 else -1
        streak[a] = _update_streak(streak[a], a_won)
        streak[b] = _update_streak(streak[b], b_won)

        ewma_form[a] = (1 - EWMA_ALPHA) * ewma_form[a] + EWMA_ALPHA * a_won
        ewma_form[b] = (1 - EWMA_ALPHA) * ewma_form[b] + EWMA_ALPHA * b_won

        toss_stats[a]["played"] += 1; toss_stats[a]["won"] += a_won_toss
        toss_stats[b]["played"] += 1; toss_stats[b]["won"] += b_won_toss
        toss_venue_stats[(a, v)]["played"] += 1; toss_venue_stats[(a, v)]["won"] += a_won_toss
        toss_venue_stats[(b, v)]["played"] += 1; toss_venue_stats[(b, v)]["won"] += b_won_toss

        # rolling ball-by-ball stats — use this match's row for each team
        for team in (a, b):
            try:
                row = tms_idx.loc[(r["match_id"], team)]
            except KeyError:
                continue
            for c in stat_cols:
                rolling[team][c].append(float(row[c]))

    feats_df = pd.DataFrame(feats)
    enriched = pd.concat([matches.reset_index(drop=True), feats_df], axis=1)

    # Final snapshot for inference-time feature lookup
    final_stats = {
        "h2h":                {str(k): dict(v) for k, v in h2h.items()},
        "venue_stats":        {f"{k[0]}||{k[1]}": dict(v) for k, v in venue_stats.items()},
        "overall_stats":      {k: dict(v) for k, v in overall_stats.items()},
        "streak":             dict(streak),
        "ewma_form":          dict(ewma_form),
        "toss_stats":         {k: dict(v) for k, v in toss_stats.items()},
        "toss_venue_stats":   {f"{k[0]}||{k[1]}": dict(v) for k, v in toss_venue_stats.items()},
        # snapshot rolling rates per team so predict.py can just look them up
        "rolling_rates": {
            team: {
                "bat_rpo":       rolling_rate(team, "bat_runs", "bat_balls"),
                "bowl_rpo":      rolling_rate(team, "bowl_runs", "bowl_balls"),
                "bat_pp_rpo":    rolling_rate(team, "bat_pp_runs", "bat_pp_balls"),
                "bowl_pp_rpo":   rolling_rate(team, "bowl_pp_runs", "bowl_pp_balls"),
                "bat_death_rpo": rolling_rate(team, "bat_death_runs", "bat_death_balls"),
                "bowl_death_rpo":rolling_rate(team, "bowl_death_runs", "bowl_death_balls"),
            }
            for team in rolling.keys()
        },
    }
    return enriched, final_stats


# SYMMETRIC AUGMENTATION 
def build_symmetric_frame(matches: pd.DataFrame, target: str) -> pd.DataFrame:
    """Two rows per match (a-vs-b, b-vs-a). Every diff flips sign in the
    second orientation so the model can't learn an a/b asymmetry bias.
    """
    rows = []
    for _, r in matches.iterrows():
        overall_diff = r["overall_winrate_a"] - r["overall_winrate_b"]
        ewma_diff    = r["ewma_form_a"] - r["ewma_form_b"]
        streak_diff  = r["streak_a"] - r["streak_b"]
        bat_diff     = r["bat_rpo_a"] - r["bat_rpo_b"]
        # Note: for bowling, LOWER rpo is better, so diff = b_rpo - a_rpo so
        # that a positive value favors team A.
        bowl_diff    = r["bowl_rpo_b"] - r["bowl_rpo_a"]
        bat_pp_diff  = r["bat_pp_rpo_a"] - r["bat_pp_rpo_b"]
        bowl_pp_diff = r["bowl_pp_rpo_b"] - r["bowl_pp_rpo_a"]
        bat_d_diff   = r["bat_death_rpo_a"] - r["bat_death_rpo_b"]
        bowl_d_diff  = r["bowl_death_rpo_b"] - r["bowl_death_rpo_a"]
        toss_rdiff   = r["toss_rate_a"] - r["toss_rate_b"]
        toss_vdiff   = r["toss_rate_a_at_venue"] - r["toss_rate_b_at_venue"]

        def row_for(orient):
            """orient = 1 means a=team1, b=team2; orient = -1 means flipped."""
            if orient == 1:
                tw_is_a = int(r["toss_winner"] == r["team1"])
                h2h_a = r["h2h_winrate_a"]
                y = int(r[target] == r["team1"])
            else:
                tw_is_a = int(r["toss_winner"] == r["team2"])
                h2h_a = 1 - r["h2h_winrate_a"]
                y = int(r[target] == r["team2"])
            return {
                "venue": r["venue"],
                "toss_decision": r["toss_decision"],
                "toss_winner_is_a":      tw_is_a,
                "h2h_winrate_a":         h2h_a,
                "overall_winrate_diff":  orient * overall_diff,
                "ewma_form_diff":        orient * ewma_diff,
                "streak_diff":           orient * streak_diff,
                "bat_rpo_diff":          orient * bat_diff,
                "bowl_rpo_diff":         orient * bowl_diff,
                "bat_pp_rpo_diff":       orient * bat_pp_diff,
                "bowl_pp_rpo_diff":      orient * bowl_pp_diff,
                "bat_death_rpo_diff":    orient * bat_d_diff,
                "bowl_death_rpo_diff":   orient * bowl_d_diff,
                "toss_rate_diff":        orient * toss_rdiff,
                "toss_rate_diff_at_venue": orient * toss_vdiff,
                "y": y,
            }

        rows.append(row_for(1))
        rows.append(row_for(-1))

    out = pd.DataFrame(rows)
    
    return out.iloc[300:].reset_index(drop=True)


# TRAIN & COMPARE 
def train_and_pick_best(X, y, label: str):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    candidates = {
        # L1 (Lasso) regularization zeroes out redundant features automatically
        
        "LR_L1_strong": LogisticRegression(
            penalty="l1", solver="liblinear", C=0.1,
            max_iter=5000, random_state=RANDOM_STATE, class_weight="balanced",
        ),
        "LR_L1_medium": LogisticRegression(
            penalty="l1", solver="liblinear", C=0.3,
            max_iter=5000, random_state=RANDOM_STATE, class_weight="balanced",
        ),
        "LR_L2":        LogisticRegression(
            penalty="l2", C=1.0,
            max_iter=5000, random_state=RANDOM_STATE, class_weight="balanced",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500, max_depth=10, min_samples_leaf=5,
            random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced",
        ),
    }
    if HAS_XGB:
        candidates["XGBoost"] = XGBClassifier(
            n_estimators=600, max_depth=4, learning_rate=0.04,
            subsample=0.9, colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=1.0,
            random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1,
            scale_pos_weight=1.0,
        )

    results, best = {}, (None, None, -1.0)
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1]
        acc = accuracy_score(y_val, preds)
        ll = log_loss(y_val, probs)
        results[name] = {"accuracy": round(acc, 4), "log_loss": round(ll, 4)}
        print(f"  [{label}] {name:20s}  acc={acc:.4f}  log_loss={ll:.4f}")
        if acc > best[2]:
            best = (name, model, acc)

    print(f"  => Best for {label}: {best[0]}  (acc={best[2]:.4f})\n")
    # Also return the training split so SHAP can use it as background data
    return best[0], best[1], results, X_train


# MAIN 
def main():
    print(f"Loading data from {DATA_PATH} ...")
    matches, team_match_stats = load_matches_and_team_stats(DATA_PATH)
    print(f"  matches: {len(matches)}  |  team-match rows: {len(team_match_stats)}")

    print("Computing time-aware features (this pass uses rolling window)...")
    enriched, final_stats = compute_historical_features(matches, team_match_stats)

    matches_2025 = matches[matches["season"] == 2025]
    all_teams = sorted(
    set(matches_2025["team1"]).union(set(matches_2025["team2"]))
)

    # all_teams = sorted(set(enriched["team1"]).union(enriched["team2"]))
    all_venues = sorted(enriched["venue"].unique())
    all_toss_decs = sorted(enriched["toss_decision"].dropna().unique())
    print(f"  teams: {len(all_teams)}, venues: {len(all_venues)}")

    encoder = FeatureEncoder()
    encoder.fit(all_teams, all_venues, all_toss_decs)

    # Match winner 
    print("\n=== Training MATCH-WINNER models ===")
    mw_frame = build_symmetric_frame(enriched, target="winner")
    X_mw = encoder.transform_match(mw_frame)
    y_mw = mw_frame["y"].values
    mw_name, mw_model, mw_results, mw_X_train = train_and_pick_best(X_mw, y_mw, "match_winner")

    # Toss winner
    print("=== Training TOSS-WINNER models ===")
    tw_frame = build_symmetric_frame(enriched, target="toss_winner")
    X_tw = encoder.transform_toss(tw_frame)
    y_tw = tw_frame["y"].values
    tw_name, tw_model, tw_results, _ = train_and_pick_best(X_tw, y_tw, "toss_winner")

    # Save artefacts
    with open(MODEL_DIR / "match_winner_model.pkl", "wb") as f:
        pickle.dump({
            "model": mw_model, "name": mw_name,
            "background": mw_X_train[:200],   # subset for SHAP explainers
        }, f)
    with open(MODEL_DIR / "toss_winner_model.pkl", "wb") as f:
        pickle.dump({"model": tw_model, "name": tw_name}, f)
    with open(MODEL_DIR / "encoders.pkl", "wb") as f:
        pickle.dump(encoder, f)
    with open(MODEL_DIR / "team_stats.pkl", "wb") as f:
        pickle.dump(final_stats, f)

    meta = {
        "teams": all_teams,
        "venues": all_venues,
        "toss_decisions": all_toss_decs,
        "match_winner_best": mw_name,
        "toss_winner_best": tw_name,
        "match_winner_results": mw_results,
        "toss_winner_results": tw_results,
        "n_matches": int(len(enriched)),
        "rolling_window": ROLLING_WINDOW,
        "ewma_alpha": EWMA_ALPHA,
    }
    with open(MODEL_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved artefacts to {MODEL_DIR}/")
    print(f"  match_winner_model.pkl  ({mw_name})")
    print(f"  toss_winner_model.pkl   ({tw_name})")
    print(f"  encoders.pkl, team_stats.pkl, meta.json")


if __name__ == "__main__":
    main()