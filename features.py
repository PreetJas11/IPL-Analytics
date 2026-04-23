"""
Shared feature-engineering module used by both train_models.py and predict.py.

"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Used by transform_match() AND by the SHAP explainer to label bars.
MATCH_NUMERIC_FEATURES = [
    "toss_winner_is_a",
    "h2h_winrate_a",
    "overall_winrate_diff",
    "ewma_form_diff",
    "streak_diff",
    "bat_rpo_diff",
    "bowl_rpo_diff",
    "bat_pp_rpo_diff",
    "bowl_pp_rpo_diff",
    "bat_death_rpo_diff",
    "bowl_death_rpo_diff",
]

TOSS_NUMERIC_FEATURES = [
    "toss_rate_diff",
    "toss_rate_diff_at_venue",
]

# Labels for the SHAP UI.
FEATURE_LABELS = {
    "toss_winner_is_a":     "Team A won the toss",
    "h2h_winrate_a":        "Head-to-head history",
    "overall_winrate_diff": "Overall win rate",
    "ewma_form_diff":       "Recent form (weighted)",
    "streak_diff":          "Current win/loss streak",
    "bat_rpo_diff":         "Batting run rate",
    "bowl_rpo_diff":        "Bowling economy",
    "bat_pp_rpo_diff":      "Powerplay batting",
    "bowl_pp_rpo_diff":     "Powerplay bowling",
    "bat_death_rpo_diff":   "Death-overs batting",
    "bowl_death_rpo_diff":  "Death-overs bowling",
    "toss_dec=bat":         "Toss decision: bat",
    "toss_dec=field":       "Toss decision: field",
}


class FeatureEncoder:

    def __init__(self):
        self.teams = []
        self.venues = []
        self.toss_decs = []

    def fit(self, all_teams, all_venues, all_toss_decs):
        self.teams = list(all_teams)
        self.venues = list(all_venues)
        self.toss_decs = list(all_toss_decs)

    @staticmethod
    def _onehot(values, categories):
        idx = {c: i for i, c in enumerate(categories)}
        out = np.zeros((len(values), len(categories)), dtype=np.float32)
        for i, v in enumerate(values):
            j = idx.get(v)
            if j is not None:
                out[i, j] = 1.0
        return out

    # Column layout of transform_match() output — needed for SHAP labels.
    def match_feature_layout(self):
        return [f"toss_dec={d}" for d in self.toss_decs] + MATCH_NUMERIC_FEATURES

    def transform_match(self, frame: pd.DataFrame) -> np.ndarray:
        toss_dec_oh = self._onehot(
            frame["toss_decision"].astype(str).tolist(), self.toss_decs
        )
        num = frame[MATCH_NUMERIC_FEATURES].astype(float).values
        return np.hstack([toss_dec_oh, num]).astype(np.float32)

    def transform_toss(self, frame: pd.DataFrame) -> np.ndarray:
        return frame[TOSS_NUMERIC_FEATURES].astype(float).values.astype(np.float32)