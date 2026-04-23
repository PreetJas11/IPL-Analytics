"""
Train the LIVE-MATCH prediction model.

While train_models.py predicts the winner BEFORE a match starts, this module
predicts the winner MID-MATCH given the current state:
    current_score, wickets, overs completed, batting_team, target (2nd inns)
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

HERE = Path(__file__).parent
MODEL_DIR = HERE / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Which features the model consumes, in order.
LIVE_FEATURES = [
    "inning",                 # 1 or 2
    "overs_done",             # 0..20 (6 balls = 1 over)
    "balls_remaining",        # 120 - balls_bowled
    "score",
    "wickets_in_hand",        # 10 - wickets
    "current_rr",             # runs per over so far
    "is_inn2",                # 0/1
    "target",                 # 0 in innings 1, else 1st-innings total + 1
    "runs_needed",            # innings-2 only, else 0
    "req_rr",                 # innings-2 only, else 0
    "rr_diff",                # current_rr - req_rr  (only meaningful in inn 2)
    "wickets_per_over",       # pressure metric
]


# dataset
def build_live_dataset(csv_path="dataset/IPLDataset.csv") -> pd.DataFrame:
    """Walk the ball-by-ball CSV and emit one row per (match, innings, over)
    snapshot. A snapshot represents match state AT THE END of that over."""
    print(f"reading {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False, usecols=[
        "match_id", "inning", "over", "ball", "batting_team", "bowling_team",
        "total_runs", "dismissal_kind", "winner", "team1", "team2", "venue", "date"
    ])
    # wicket bool: dismissal_kind is "not out" for non-wicket balls
    df["is_wkt"] = (df["dismissal_kind"].astype(str) != "not out").astype(int)

    # Keep only matches with a decided winner
    m_ids = df.groupby("match_id").agg(
        team1=("team1", "first"),
        team2=("team2", "first"),
        winner=("winner", "first"),
    ).reset_index()
    m_ids = m_ids.dropna(subset=["winner", "team1", "team2"])
    m_ids = m_ids[m_ids.apply(lambda r: r["winner"] in (r["team1"], r["team2"]), axis=1)]
    df = df[df["match_id"].isin(m_ids["match_id"])]

    print(f"  {df['match_id'].nunique()} matches, {len(df):,} balls")

    rows = []
    for mid, g in df.groupby("match_id", sort=False):
        winner = g["winner"].iloc[0]
        venue  = g["venue"].iloc[0]
        team1, team2 = g["team1"].iloc[0], g["team2"].iloc[0]

        inn1 = g[g["inning"] == 1]
        target = int(inn1["total_runs"].sum()) + 1 if len(inn1) else 0

        for inning in (1, 2):
            inn = g[g["inning"] == inning].sort_values(["over", "ball"]).reset_index(drop=True)
            if len(inn) == 0:
                continue
            bt = inn["batting_team"].iloc[0]
            bl = inn["bowling_team"].iloc[0]

            cum_runs = inn["total_runs"].cumsum().values
            cum_wkts = inn["is_wkt"].cumsum().values
            n_balls = len(inn)

            # Snapshot at end of each completed over (6, 12, 18, ..., 120 balls)
            # Also include an early snapshot at 3 balls (half-over) for signal.
            for balls_bowled in [3] + list(range(6, 121, 6)):
                idx = balls_bowled - 1
                if idx >= n_balls:
                    break
                score = int(cum_runs[idx])
                wkts  = int(cum_wkts[idx])
                overs_done = balls_bowled / 6.0

                is_inn2 = int(inning == 2)
                runs_needed = max(0, target - score) if is_inn2 else 0
                balls_remaining = max(0, 120 - balls_bowled)
                req_rr = (runs_needed * 6.0 / balls_remaining) if (is_inn2 and balls_remaining > 0) else 0.0
                current_rr = score / overs_done if overs_done > 0 else 0.0
                rr_diff = current_rr - req_rr if is_inn2 else 0.0

                rows.append({
                    "match_id": int(mid),
                    "inning": inning,
                    "overs_done": overs_done,
                    "balls_remaining": balls_remaining,
                    "score": score,
                    "wickets_in_hand": 10 - wkts,
                    "current_rr": current_rr,
                    "is_inn2": is_inn2,
                    "target": target if is_inn2 else 0,
                    "runs_needed": runs_needed,
                    "req_rr": req_rr,
                    "rr_diff": rr_diff,
                    "wickets_per_over": wkts / overs_done if overs_done > 0 else 0.0,
                    # label: batting team wins?
                    "y": int(winner == bt),
                    # metadata for inspection
                    "batting_team": bt, "bowling_team": bl,
                    "venue": venue, "winner": winner,
                })

    out = pd.DataFrame(rows)
    print(f"  built {len(out):,} live snapshots")
    return out


#  model
def train(snapshots: pd.DataFrame) -> dict:
    X = snapshots[LIVE_FEATURES].values.astype(np.float32)
    y = snapshots["y"].values.astype(int)

   
    match_ids = snapshots["match_id"].unique()
    rng = np.random.default_rng(42)
    rng.shuffle(match_ids)
    n_test = max(1, int(len(match_ids) * 0.2))
    test_ids = set(match_ids[:n_test])
    test_mask  = snapshots["match_id"].isin(test_ids).values
    train_mask = ~test_mask

    X_tr, X_te = X[train_mask], X[test_mask]
    y_tr, y_te = y[train_mask], y[test_mask]

    candidates = {
        "LR_L2":  LogisticRegression(max_iter=2000, class_weight="balanced"),
        "LR_L1":  LogisticRegression(max_iter=2000, penalty="l1", solver="liblinear",
                                     C=0.3, class_weight="balanced"),
    }
    # XGBoost if available
    try:
        import xgboost as xgb
        candidates["XGBoost"] = xgb.XGBClassifier(
            max_depth=4, n_estimators=300, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            objective="binary:logistic", eval_metric="logloss",
            verbosity=0,
        )
    except Exception:
        print("  xgboost not available — skipping")

    results = {}
    best_name, best_model, best_ll = None, None, np.inf
    for name, model in candidates.items():
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]
        acc = accuracy_score(y_te, (p >= 0.5).astype(int))
        ll  = log_loss(y_te, np.clip(p, 1e-6, 1 - 1e-6))
        results[name] = {"accuracy": round(float(acc), 4),
                         "log_loss": round(float(ll), 4)}
        print(f"  {name:10s}  acc={acc:.4f}  log_loss={ll:.4f}")
        if ll < best_ll:
            best_ll, best_name, best_model = ll, name, model

    print(f"  best model: {best_name}")
    return {
        "model": best_model, "name": best_name, "results": results,
        "features": LIVE_FEATURES,
        "background": X_tr[:500],
        "n_snapshots": int(len(snapshots)),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
    }


#  main
def main():
    snaps = build_live_dataset()
    print("\n=== Training LIVE-MATCH model ===")
    art = train(snaps)

    with open(MODEL_DIR / "live_match_model.pkl", "wb") as f:
        pickle.dump(art, f)
    print(f"\nsaved models/live_match_model.pkl")

    latest_season = int(snaps.merge(
        pd.read_csv("dataset/IPLDataset.csv", usecols=["match_id", "season"]).drop_duplicates("match_id"),
        on="match_id", how="left",
    )["season"].max())
    df_latest = pd.read_csv("dataset/IPLDataset.csv", usecols=["season", "team1", "team2"], low_memory=False)
    df_latest = df_latest[df_latest["season"] == latest_season]
    current_teams = sorted(set(df_latest["team1"]).union(df_latest["team2"]))
    print(f"  current season: {latest_season} · {len(current_teams)} teams")

    # Merge into meta.json so dashboards can see it
    meta_path = MODEL_DIR / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta["live_match_best"] = art["name"]
    meta["live_match_results"] = art["results"]
    meta["live_match_features"] = art["features"]
    meta["live_match_n_snapshots"] = art["n_snapshots"]
    meta["current_season"] = latest_season
    meta["current_season_teams"] = current_teams
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"updated {meta_path}")


if __name__ == "__main__":
    main()