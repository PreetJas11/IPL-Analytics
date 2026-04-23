
"""
IPL Dashboard — Match & Toss Winner Prediction (with SHAP + live)

"""

import altair as alt
import pandas as pd
import streamlit as st

import predict  # inference module


# helpers
def _prob_bar(team_a, team_b, prob_a, prob_b):
    """Horizontal stacked probability bar for two teams."""
    df = pd.DataFrame({
        "Team": [team_a, team_b],
        "Win Probability %": [round(prob_a * 100, 2), round(prob_b * 100, 2)],
    })
    st.bar_chart(df.set_index("Team"), horizontal=True, height=180)


def _shap_bar(contribs, team_a, team_b, top_n=7):
    """Horizontal SHAP bar chart: one bar per feature, colored by which team
    the model thinks it favors. Positive SHAP -> pushes probability toward A."""
    if not contribs:
        st.info("No SHAP contributions available for this prediction.")
        return

    rows = [{
        "Feature":        c["label"],
        "Impact (logit)": c["shap"],
        "Favors":         team_a if c["favors"] == "team_a" else team_b,
        "Value":          c["value"],
    } for c in contribs[:top_n]]
    df = pd.DataFrame(rows)

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Impact (logit):Q",
                    title=f"<- favors {team_b}     |     favors {team_a} ->"),
            y=alt.Y("Feature:N", sort="-x", title=None),
            color=alt.Color(
                "Favors:N",
                scale=alt.Scale(domain=[team_a, team_b],
                                range=["#1f77b4", "#ff7f0e"]),
                legend=alt.Legend(title="Pushes prediction toward"),
            ),
            tooltip=["Feature", "Value", "Impact (logit)", "Favors"],
        )
        .properties(height=max(180, 32 * len(df)))
    )
    st.altair_chart(chart, use_container_width=True)


# Live-match block 
@st.fragment
def _live_block(team_a: str, team_b: str, venue: str):
    live_info = predict.live_model_info()
    if not live_info:
        st.warning(
            "Live-match model not found. Run `python train_live_model.py` "
            "in the outputs folder, then reload this tab."
        )
        return

    acc = live_info["results"][live_info["name"]]["accuracy"]
    st.caption(
        f"Live model: **{live_info['name']}** · "
        f"{acc * 100:.1f}% accuracy on {live_info['n_snapshots']:,} "
        "historical over-by-over snapshots."
    )

    # state inputs
    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        inning = st.radio("Innings", [1, 2], horizontal=True, key="live_inning")
    with lc2:
        batting_team = st.selectbox("Batting team", [team_a, team_b], key="live_bat")
    with lc3:
        overs_done = st.number_input(
            "Overs completed", min_value=0.0, max_value=20.0,
            value=10.0, step=0.1, key="live_overs",
            help="5.3 = 5 overs and 3 balls",
        )

    lc4, lc5, lc6 = st.columns(3)
    with lc4:
        current_score = st.number_input(
            "Current score", min_value=0, max_value=400, value=80, key="live_score"
        )
    with lc5:
        wkts = st.number_input(
            "Wickets fallen", min_value=0, max_value=10, value=2, key="live_wkts"
        )
    with lc6:
        target = st.number_input(
            "Target (innings 2 only)", min_value=0, max_value=400,
            value=0 if inning == 1 else 180,
            key="live_target", disabled=(inning == 1),
        )

    if st.button("Predict live", key="live_go", use_container_width=True):
        st.session_state["live_has_predicted"] = True

    
    if st.session_state.get("live_has_predicted"):
        try:
            live_res = predict.predict_live_match(
                team_a, team_b, venue,
                current_score=int(current_score),
                wickets=int(wkts),
                overs=float(overs_done),
                batting_team=batting_team,
                target=int(target) if inning == 2 else None,
                inning=int(inning),
            )
        except Exception as e:
            st.error(f"Live prediction failed: {e}")
            return

        lm1, lm2 = st.columns(2)
        lm1.metric(f"{team_a} — win %", f"{live_res['prob_a'] * 100:.1f}%")
        lm2.metric(f"{team_b} — win %", f"{live_res['prob_b'] * 100:.1f}%")
        _prob_bar(team_a, team_b, live_res["prob_a"], live_res["prob_b"])
        st.success(live_res["commentary"])

        # with st.expander("State fed to the live model"):
        #     state_df = pd.DataFrame(
        #         [{"Feature": k, "Value": round(v, 3) if isinstance(v, float) else v}
        #          for k, v in live_res["state"].items()]
        #     )
        #     st.dataframe(state_df, hide_index=True, use_container_width=True)


# main render 
def render():
    st.header("Momentum Match Predictor")
    info = predict.model_info()
    season = predict.current_season()
    # caption = (
    #     f"Trained on {info['n_matches_trained_on']} IPL matches · "
    #     f"Match model: **{info['match_winner_model']}** · "
    #     f"Toss model: **{info['toss_winner_model']}**"
    # )
    # if season:
    #     caption += f" · Showing teams from **{season}** season"
    # st.caption(caption)

    # Only current-season franchises, not the full historical list.
    teams = predict.list_active_teams()
    venues = predict.list_venues()

    # Inputs 
    c1, c2, c3,c4 = st.columns([1, 1, 1.2, 0.6])
    with c1:
        team_a = st.selectbox("Team A", teams, index=0, key="pred_team_a")
    with c2:
        team_b_opts = [t for t in teams if t != team_a]
        team_b = st.selectbox(
            "Team B", team_b_opts,
            index=0 if team_b_opts else 0,
            key="pred_team_b",
        )
    with c3:
        venue = st.selectbox("Venue", venues, key="pred_venue")

    with c4:
        st.write("")
        st.write("")
        predict_clicked = st.button("Predict", type="primary", use_container_width=True)
    
    if predict_clicked:
        st.session_state.has_predicted = True



    with st.expander("Advanced: provide toss info (optional)"):
        toss_winner_label = st.radio(
            "Toss winner",
            options=["Unknown (predict it)", team_a, team_b],
            horizontal=True, key="pred_toss_winner",
        )
        toss_decision = st.radio(
            "Toss decision",
            options=["Unknown"] + predict.list_toss_decisions(),
            horizontal=True, key="pred_toss_decision",
        )

    
   
    st.session_state.setdefault("has_predicted", False)

    if not st.session_state.has_predicted:
        st.info("Pick two teams and a venue.")
        return

    if team_a == team_b:
        st.error("Pick two *different* teams.")
        return

    col_toss, col_match= st.columns(2)
    with col_toss: # Toss prediction
        st.subheader("Toss winner")
        toss_res = predict.predict_toss_winner(team_a, team_b, venue)
        tc1, tc2 = st.columns(2)
        tc1.metric(f"{team_a} — toss win %", f"{toss_res['prob_a'] * 100:.1f}%")
        tc2.metric(f"{team_b} — toss win %", f"{toss_res['prob_b'] * 100:.1f}%")
        _prob_bar(team_a, team_b, toss_res["prob_a"], toss_res["prob_b"])

    with col_match: #Match prediction
        st.subheader("Match winner")
        tw = None if toss_winner_label.startswith("Unknown") else toss_winner_label
        td = None if toss_decision == "Unknown" else toss_decision

        explanation = predict.explain_prediction(
            team_a, team_b, venue, toss_winner=tw, toss_decision=td
        )
        match_res = explanation["prediction"]

        mc1, mc2 = st.columns(2)
        mc1.metric(f"{team_a} — win %", f"{match_res['prob_a'] * 100:.1f}%")
        mc2.metric(f"{team_b} — win %", f"{match_res['prob_b'] * 100:.1f}%")
        _prob_bar(team_a, team_b, match_res["prob_a"], match_res["prob_b"])

        winner = team_a if match_res["prob_a"] > match_res["prob_b"] else team_b
        confidence = max(match_res["prob_a"], match_res["prob_b"]) * 100
    st.success(f"**Predicted winner:** {winner}  ({confidence:.1f}% chances)")

    # if match_res["toss_winner_used"] == "marginalised":
    #     st.caption(
    #         "Toss info wasn't provided, so the match probability was averaged "
    #     )

    # SHAP explanation
    st.subheader("Where did the match tilt?")
    _shap_bar(explanation["contributions"], team_a, team_b)

    st.markdown("### Match Win Summary")
    st.markdown(explanation["summary"])

    # with st.expander("Feature values used for this prediction"):
    #     val_rows = [
    #         {"Feature":     c["label"],
    #          "Value":       round(c["value"], 3),
    #          "SHAP impact": round(c["shap"], 4),
    #          "Favors":      team_a if c["favors"] == "team_a" else team_b}
    #         for c in explanation["contributions"]
    #     ]
    #     if val_rows:
    #         st.dataframe(pd.DataFrame(val_rows), hide_index=True, use_container_width=True)
    #     else:
    #         st.write("No non-zero features (model predicted near base rate).")

    # ---- Live match prediction (isolated fragment) -------------------------
    with st.expander("🏏 Live match prediction", expanded=False):
        _live_block(team_a, team_b, venue)


# Allow standalone run for quick testing:  streamlit run tab7_prediction.py
if __name__ == "__main__":
    st.set_page_config(page_title="IPL Prediction", layout="wide")
    render()