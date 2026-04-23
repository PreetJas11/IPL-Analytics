
"""
IPL Prediction — Inference Module 
Loads models + end-of-training stats snapshot, exposes results

"""

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import shap

from features import (
    MATCH_NUMERIC_FEATURES, TOSS_NUMERIC_FEATURES, FEATURE_LABELS
)

HERE = Path(__file__).parent
MODEL_DIR = HERE / "models"


# Load artefacts
def _load():
    with open(MODEL_DIR / "match_winner_model.pkl", "rb") as f:
        mw = pickle.load(f)
    with open(MODEL_DIR / "toss_winner_model.pkl", "rb") as f:
        tw = pickle.load(f)
    with open(MODEL_DIR / "encoders.pkl", "rb") as f:
        enc = pickle.load(f)
    with open(MODEL_DIR / "team_stats.pkl", "rb") as f:
        stats = pickle.load(f)
    with open(MODEL_DIR / "meta.json") as f:
        meta = json.load(f)
    live = None
    live_path = MODEL_DIR / "live_match_model.pkl"
    if live_path.exists():
        with open(live_path, "rb") as f:
            live = pickle.load(f)
    return mw, tw, enc, stats, meta, live


_MW, _TW, _ENC, _STATS, _META, _LIVE = _load()


# Build the SHAP explainer 
def _build_explainer():
    model = _MW["model"]
    background = _MW.get("background")
    try:
        if hasattr(model, "coef_"):
            return shap.LinearExplainer(model, background)
        return shap.Explainer(model, background)
    except Exception as e:
        print(f"[warn] SHAP explainer init failed: {e}")
        return None


_EXPLAINER = _build_explainer()


# Public helpers 
def list_teams():          return list(_META["teams"])
def list_venues():         return list(_META["venues"])
def list_toss_decisions(): return list(_META["toss_decisions"])


def list_active_teams():
    return list(_META.get("current_season_teams") or _META["teams"])


def current_season():
    return _META.get("current_season")


def model_info():
    return {
        "match_winner_model": _META["match_winner_best"],
        "toss_winner_model":  _META["toss_winner_best"],
        "n_matches_trained_on": _META["n_matches"],
        "match_winner_results": _META["match_winner_results"],
        "toss_winner_results":  _META["toss_winner_results"],
    }


# Historical stat lookups
def _winrate(d):
    return (d["won"] / d["played"]) if d and d["played"] else 0.5


def _h2h_winrate(a, b):
    key = str(tuple(sorted([a, b])))
    d = _STATS["h2h"].get(key)
    if not d or d["played"] == 0:
        return 0.5
    first = sorted([a, b])[0]
    a_wins = d["a_won"] if first == a else d["played"] - d["a_won"]
    return a_wins / d["played"]


def _venue_winrate(team, venue):   return _winrate(_STATS["venue_stats"].get(f"{team}||{venue}"))
def _overall_winrate(team):        return _winrate(_STATS["overall_stats"].get(team))
def _toss_rate(team):              return _winrate(_STATS.get("toss_stats", {}).get(team))
def _toss_rate_venue(team, venue): return _winrate(_STATS.get("toss_venue_stats", {}).get(f"{team}||{venue}"))
def _streak(team):                 return int(_STATS.get("streak", {}).get(team, 0))
def _ewma_form(team):              return float(_STATS.get("ewma_form", {}).get(team, 0.5))


def _rolling(team, key):
    return float(_STATS.get("rolling_rates", {}).get(team, {}).get(key, 0.0))


def _historical_features(team_a, team_b, venue):
    """Returns exactly the diff-features the match + toss models expect,
    plus raw values for the UI / explanations.
    """
    bat_diff         = _rolling(team_a, "bat_rpo")        - _rolling(team_b, "bat_rpo")
    bowl_diff        = _rolling(team_b, "bowl_rpo")       - _rolling(team_a, "bowl_rpo")   # sign flipped: + favors a
    bat_pp_diff      = _rolling(team_a, "bat_pp_rpo")     - _rolling(team_b, "bat_pp_rpo")
    bowl_pp_diff     = _rolling(team_b, "bowl_pp_rpo")    - _rolling(team_a, "bowl_pp_rpo")
    bat_death_diff   = _rolling(team_a, "bat_death_rpo")  - _rolling(team_b, "bat_death_rpo")
    bowl_death_diff  = _rolling(team_b, "bowl_death_rpo") - _rolling(team_a, "bowl_death_rpo")
    return {
        # match features
        "h2h_winrate_a":         _h2h_winrate(team_a, team_b),
        "overall_winrate_diff":  _overall_winrate(team_a) - _overall_winrate(team_b),
        "ewma_form_diff":        _ewma_form(team_a) - _ewma_form(team_b),
        "streak_diff":           _streak(team_a) - _streak(team_b),
        "bat_rpo_diff":          bat_diff,
        "bowl_rpo_diff":         bowl_diff,
        "bat_pp_rpo_diff":       bat_pp_diff,
        "bowl_pp_rpo_diff":      bowl_pp_diff,
        "bat_death_rpo_diff":    bat_death_diff,
        "bowl_death_rpo_diff":   bowl_death_diff,
        # toss-model features
        "toss_rate_diff":          _toss_rate(team_a) - _toss_rate(team_b),
        "toss_rate_diff_at_venue": _toss_rate_venue(team_a, venue) - _toss_rate_venue(team_b, venue),
        # raw values for display
        "_overall_a": _overall_winrate(team_a),  "_overall_b": _overall_winrate(team_b),
        "_ewma_a":    _ewma_form(team_a),        "_ewma_b":    _ewma_form(team_b),
        "_streak_a":  _streak(team_a),           "_streak_b":  _streak(team_b),
        "_bat_rpo_a": _rolling(team_a, "bat_rpo"),  "_bat_rpo_b":  _rolling(team_b, "bat_rpo"),
        "_bowl_rpo_a":_rolling(team_a, "bowl_rpo"), "_bowl_rpo_b": _rolling(team_b, "bowl_rpo"),
    }


# Core prediction 
def _build_match_row(team_a, team_b, venue, toss_winner_is_a, toss_decision):
    feats = _historical_features(team_a, team_b, venue)
    return {
        "team_a": team_a, "team_b": team_b, "venue": venue,
        "toss_decision": toss_decision,
        "toss_winner_is_a": int(toss_winner_is_a),
        **{k: feats[k] for k in MATCH_NUMERIC_FEATURES if k in feats},
    }


def _match_proba(team_a, team_b, venue, toss_winner_is_a, toss_decision):
    frame = pd.DataFrame([_build_match_row(team_a, team_b, venue,
                                           toss_winner_is_a, toss_decision)])
    X = _ENC.transform_match(frame)
    return float(_MW["model"].predict_proba(X)[0, 1])


def predict_toss_winner(team_a, team_b, venue) -> dict:
    row = {"team_a": team_a, "team_b": team_b, "venue": venue,
           **_historical_features(team_a, team_b, venue)}
    frame = pd.DataFrame([row])
    X = _ENC.transform_toss(frame)
    prob_a = float(_TW["model"].predict_proba(X)[0, 1])
    return {"team_a": team_a, "team_b": team_b,
            "prob_a": prob_a, "prob_b": 1 - prob_a}


def predict_match_winner(team_a, team_b, venue,
                         toss_winner: Optional[str] = None,
                         toss_decision: Optional[str] = None) -> dict:
    if toss_winner in (team_a, team_b) and toss_decision:
        p_a = _match_proba(team_a, team_b, venue,
                           toss_winner == team_a, toss_decision)
        return {"team_a": team_a, "team_b": team_b,
                "prob_a": p_a, "prob_b": 1 - p_a,
                "toss_winner_used": toss_winner,
                "toss_decision_used": toss_decision}

    # Marginalise unknown toss using the toss model + uniform prior on decision
    tp = predict_toss_winner(team_a, team_b, venue)
    decisions = list_toss_decisions()
    p_a_total = 0.0
    for tw_is_a, p_tw in [(True, tp["prob_a"]), (False, tp["prob_b"])]:
        for dec in decisions:
            p_a_total += p_tw * (1.0 / len(decisions)) * _match_proba(
                team_a, team_b, venue, tw_is_a, dec
            )
    return {"team_a": team_a, "team_b": team_b,
            "prob_a": p_a_total, "prob_b": 1 - p_a_total,
            "toss_winner_used": "marginalised",
            "toss_decision_used": "marginalised"}


# SHAP explanation 
def explain_prediction(team_a, team_b, venue,
                       toss_winner: Optional[str] = None,
                       toss_decision: Optional[str] = None) -> dict:
    """Returns per-feature SHAP contributions for one prediction.

    Contributions are in logit-space (additive on top of the base rate).
    A positive SHAP value pushes probability toward team_a.
    """
    prediction = predict_match_winner(team_a, team_b, venue, toss_winner, toss_decision)

    # For SHAP we need a concrete (toss_winner, toss_decision) — marginalised
    if prediction["toss_winner_used"] == "marginalised":
        tp = predict_toss_winner(team_a, team_b, venue)
        tw_is_a = tp["prob_a"] >= tp["prob_b"]
        toss_dec = list_toss_decisions()[0]   # use first (usually "bat")
    else:
        tw_is_a = (prediction["toss_winner_used"] == team_a)
        toss_dec = prediction["toss_decision_used"]

    row = _build_match_row(team_a, team_b, venue, tw_is_a, toss_dec)
    frame = pd.DataFrame([row])
    X = _ENC.transform_match(frame)
    layout = _ENC.match_feature_layout()

    # Compute SHAP values
    if _EXPLAINER is None:
        return {"prediction": prediction, "contributions": [], "summary": ""}

    try:
        sv = _EXPLAINER(X)
        # shap_values array: shape (1, n_features) or (1, n_features, n_classes)
        vals = np.array(sv.values).squeeze()
        if vals.ndim == 2:    # e.g. (n_features, 2) for a 2-class tree model
            vals = vals[:, 1] # class 1 = team_a wins
    except Exception as e:
        print(f"[warn] SHAP failed: {e}")
        return {"prediction": prediction, "contributions": [], "summary": ""}

    # Assemble contributions list, sorted by absolute magnitude
    contribs = []
    raw_values = np.array(X[0], dtype=float)
    for feat_name, val, shap_val in zip(layout, raw_values, vals):
        if abs(shap_val) < 1e-4:
            continue
        contribs.append({
            "feature": feat_name,
            "label": FEATURE_LABELS.get(feat_name, feat_name),
            "value": float(val),
            "shap": float(shap_val),
            "favors": "team_a" if shap_val > 0 else "team_b",
        })
    contribs.sort(key=lambda c: -abs(c["shap"]))

    summary = _build_summary(team_a, team_b, prediction, contribs, row)
    return {"prediction": prediction, "contributions": contribs, "summary": summary}


# *** Natural-language summary ***
def _describe_factor(c, team_a, team_b, row):
    """Factual description of a single contribution.
    """
    feat = c["feature"]
    v = c["value"]

    if feat == "overall_winrate_diff":
        higher = team_a if v > 0 else team_b
        return f"{higher} has a {abs(v) * 100:.0f}pp higher career win rate"
    if feat == "ewma_form_diff":
        better = team_a if v > 0 else team_b
        return f"{better} has stronger recent form (EWMA)"
    if feat == "streak_diff":
        if v == 0:
            return "both teams have neutral streaks"
        stronger = team_a if v > 0 else team_b
        return f"{stronger} comes in on a stronger streak (gap of {abs(v):.0f})"
    if feat == "h2h_winrate_a":
        if v > 0.5:
            return f"{team_a} leads the head-to-head ({v * 100:.0f}%)"
        if v < 0.5:
            return f"{team_b} leads the head-to-head ({(1 - v) * 100:.0f}%)"
        return "head-to-head is even"
    if feat == "bat_death_rpo_diff":
        who = team_a if v > 0 else team_b
        return f"{who} bats {abs(v):.2f} rpo stronger in the death overs"
    if feat == "bowl_death_rpo_diff":
        who = team_a if v > 0 else team_b
        return f"{who} bowls {abs(v):.2f} rpo tighter at the death"
    if feat == "bat_pp_rpo_diff":
        who = team_a if v > 0 else team_b
        return f"{who} bats {abs(v):.2f} rpo stronger in the powerplay"
    if feat == "bowl_pp_rpo_diff":
        who = team_a if v > 0 else team_b
        return f"{who} bowls {abs(v):.2f} rpo tighter in the powerplay"
    if feat == "bat_rpo_diff":
        who = team_a if v > 0 else team_b
        return f"{who} has a {abs(v):.2f} rpo higher overall batting rate"
    if feat == "bowl_rpo_diff":
        who = team_a if v > 0 else team_b
        return f"{who} has a {abs(v):.2f} rpo tighter overall bowling economy"
    if feat == "toss_winner_is_a":
        winner = team_a if row.get("toss_winner_is_a") else team_b
        return f"{winner} won the toss"
    if feat.startswith("toss_dec="):
        return f"Toss decision: {feat.split('=')[1]}"
    return f"{c['label']} (value {v:+.2f})"


def _build_summary(team_a, team_b, prediction, contribs, row) -> str:
    """Markdown summary: predicted winner, top pro factors, top con factors."""
    winner = team_a if prediction["prob_a"] > prediction["prob_b"] else team_b
    loser  = team_b if winner == team_a else team_a
    conf = max(prediction["prob_a"], prediction["prob_b"]) * 100
    if not contribs:
        return f"**{winner}** is predicted to win ({conf:.0f}% confidence)."

    pro = [c for c in contribs if (c["favors"] == "team_a") == (winner == team_a)][:3]
    con = [c for c in contribs if (c["favors"] == "team_a") != (winner == team_a)][:2]

    lines = [f"**{winner}** is predicted to win ({conf:.0f}% confidence)."]
    if pro:
        lines.append("")
        lines.append(f"**Signals pushing toward {winner}:**")
        for c in pro:
            lines.append(f"- {_describe_factor(c, team_a, team_b, row)}")
    if con:
        lines.append("")
        lines.append(f"**Counter-signals for {loser}:**")
        for c in con:
            lines.append(f"- {_describe_factor(c, team_a, team_b, row)}")
    return "\n".join(lines)


# Live-match prediction
def live_model_info():
    """Info about the live-match model, or None if it hasn't been trained."""
    if _LIVE is None:
        return None
    return {
        "name":      _LIVE.get("name"),
        "results":   _LIVE.get("results"),
        "features":  _LIVE.get("features"),
        "n_snapshots": _LIVE.get("n_snapshots"),
    }


def _build_live_features(inning: int, overs: float, score: int, wickets: int,
                         target: int = 0) -> dict:
    """Given the live match state, derive the feature vector the live
    model was trained on. Shared by predict_live_match() and explain_live()."""
    balls_bowled = int(round(overs * 6))
    balls_remaining = max(0, 120 - balls_bowled)
    overs_done = max(0.0, float(overs))
    is_inn2 = int(inning == 2)
    wickets = max(0, min(10, int(wickets)))
    score = max(0, int(score))
    target = max(0, int(target)) if is_inn2 else 0

    current_rr = (score / overs_done) if overs_done > 0 else 0.0
    runs_needed = max(0, target - score) if is_inn2 else 0
    req_rr = (runs_needed * 6.0 / balls_remaining) if (is_inn2 and balls_remaining > 0) else 0.0
    rr_diff = (current_rr - req_rr) if is_inn2 else 0.0
    wkt_per_over = (wickets / overs_done) if overs_done > 0 else 0.0

    return {
        "inning":            inning,
        "overs_done":        overs_done,
        "balls_remaining":   balls_remaining,
        "score":             score,
        "wickets_in_hand":   10 - wickets,
        "current_rr":        current_rr,
        "is_inn2":           is_inn2,
        "target":            target,
        "runs_needed":       runs_needed,
        "req_rr":            req_rr,
        "rr_diff":           rr_diff,
        "wickets_per_over":  wkt_per_over,
    }


def predict_live_match(team_a: str, team_b: str, venue: str,
                       current_score: int, wickets: int, overs: float,
                       batting_team: str,
                       target: Optional[int] = None,
                       inning: Optional[int] = None) -> dict:
    
    if _LIVE is None:
        raise RuntimeError(
            "Live-match model not loaded. Run `python train_live_model.py` "
            "to train it and produce models/live_match_model.pkl."
        )
    if batting_team not in (team_a, team_b):
        raise ValueError(f"batting_team must be one of {team_a!r} or {team_b!r}")

    # Infer innings if not supplied
    if inning is None:
        inning = 2 if (target and target > 0) else 1
    if inning not in (1, 2):
        raise ValueError("inning must be 1 or 2")

    feats = _build_live_features(inning, overs, current_score, wickets,
                                 target or 0)
    order = _LIVE["features"]
    x = np.array([[feats[k] for k in order]], dtype=np.float32)

    prob_batting_wins = float(_LIVE["model"].predict_proba(x)[0, 1])

    if batting_team == team_a:
        prob_a, prob_b = prob_batting_wins, 1 - prob_batting_wins
    else:
        prob_a, prob_b = 1 - prob_batting_wins, prob_batting_wins

    # A concise natural-language read of where the match is
    commentary = _live_commentary(team_a, team_b, batting_team, feats,
                                  prob_batting_wins)

    return {
        "team_a": team_a, "team_b": team_b,
        "prob_a": prob_a, "prob_b": prob_b,
        "batting_team": batting_team,
        "bowling_team": team_b if batting_team == team_a else team_a,
        "inning": inning,
        "state": feats,
        "commentary": commentary,
        "model_name": _LIVE.get("name"),
    }


def _live_commentary(team_a, team_b, batting_team, feats, prob_batting) -> str:
    """Short human-readable match-situation sentence."""
    bt = batting_team
    other = team_b if batting_team == team_a else team_a
    conf = max(prob_batting, 1 - prob_batting) * 100
    fav = bt if prob_batting > 0.5 else other

    overs = feats["overs_done"]
    score = feats["score"]
    wih = feats["wickets_in_hand"]

    if feats["is_inn2"]:
        need = feats["runs_needed"]
        rr  = feats["req_rr"]
        if need <= 0:
            state = f"{bt} has already chased down the target."
        else:
            state = (f"{bt} need {need} from {feats['balls_remaining']} balls "
                     f"(required rate {rr:.2f}) with {wih} wickets in hand.")
    else:
        state = (f"{bt} are {score}/{10 - wih} after {overs:.1f} overs "
                 f"(current rate {feats['current_rr']:.2f}).")

    return f"{state} This gives {fav} a {conf:.0f}% chance of winning."


if __name__ == "__main__":
    import pprint
    pprint.pp(model_info())
    print()
    e = explain_prediction("Mumbai Indians", "Chennai Super Kings",
                           "Wankhede Stadium")
    print("PREDICTION:", e["prediction"])
    print()
    print("CONTRIBUTIONS:")
    for c in e["contributions"]:
        print(f"  {c['label']:30s}  value={c['value']:+.3f}  shap={c['shap']:+.4f}  ({c['favors']})")
    print()
    print("SUMMARY:")
    print(e["summary"])