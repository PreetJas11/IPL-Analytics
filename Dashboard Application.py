import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="IPL Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# LOAD DATA
df = pd.read_csv("dataset/IPL_dataset.csv")
df["season"] = df["season"].astype(int)

# SIDEBAR FILTERS
st.sidebar.title(" Explore IPL match data.")

seasons = sorted(df["season"].unique().tolist())

selected_season_range = st.sidebar.slider(
    "📅 Season Range",
    min_value=min(seasons),
    max_value=max(seasons),
    value=(min(seasons), max(seasons))
)

df = df[df["season"].between(
    selected_season_range[0],
    selected_season_range[1]
)]

teams = sorted(pd.unique(df[["batting_team", "bowling_team"]].values.ravel("K")))
teams = [t for t in teams if pd.notna(t)]

selected_team = st.sidebar.selectbox(
    "🏏 Select Team",
    options=["All"] + teams
)

if selected_team != "All":
    df = df[
        (df["batting_team"] == selected_team) |
        (df["bowling_team"] == selected_team)
    ]


st.sidebar.markdown("""
### Explore IPL match data through interactive filters.

• 2007 to 2025 IPL Seasons  
• Ball-by-ball match data  
• Team-wise performance tracking  
""")

# HEADER
st.markdown("""
<h1 style='text-align: center; color: #FF4B4B;'>
🏏 IPL Analytics Hub: 18 Seasons of Insights
</h1>
<h5 style='text-align: center; color: gray;'>
Exploring IPL matches from 2007 to 2025, analyzing trends, performance, and outcomes across seasons.
</h5>
""", unsafe_allow_html=True)

st.divider()

# TAB LAYOUT
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "IPL Overview",
    "Batting Stats",
    "Bowling Performance",
    "Game Outcome",
    "Game Changers",
    "Teams Hub"
])

# =========================================================
# 🏠 TAB 1 - OVERVIEW
# =========================================================
with tab1:

    st.markdown("High-level structural view of IPL seasons, scoring patterns, and match setup")

    st.divider()

    # KPI SECTION
    total_matches = df["match_id"].nunique()
    total_runs = df["total_runs"].sum()
    total_wickets = df["is_wicket"].sum()
    avg_runs_per_match = total_runs / total_matches if total_matches else 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("🏏 Matches", f"{total_matches:,}")
    col2.metric("🏃 Total Runs", f"{total_runs:,}")
    col3.metric("🎯 Total Wickets", f"{total_wickets:,}")
    col4.metric("⚡ Avg Runs/Match", f"{round(avg_runs_per_match, 2):,}")

    st.divider()

    ##  Season Trends Overview"

    # ROW 1
    row1_col1, row1_col2 = st.columns(2)

    # 📊 Matches per Season
    with row1_col1:
        st.markdown("Matches Played per Season")

        matches_per_season = df.groupby("season")["match_id"].nunique().sort_index()

        fig, ax = plt.subplots(figsize=(6, 4))

        colors = plt.cm.viridis(range(len(matches_per_season)))

        bars = ax.bar(
            matches_per_season.index.astype(str),
            matches_per_season.values,
            color=colors
        )

        # ax.set_title("Matches Played per Season")
        ax.set_xlabel("Season")
        ax.set_ylabel("Number of Matches")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.xticks(rotation=45)

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                int(height),
                ha='center',
                va='bottom',
                fontsize=8
            )

        st.pyplot(fig)

    #  Avg Runs per Season
    with row1_col2:
        st.markdown("Average Runs per Season")

        season_runs = df.groupby(["season", "match_id"])["total_runs"].sum().reset_index()
        season_avg_runs = season_runs.groupby("season")["total_runs"].mean()
        season_avg_runs.index = season_avg_runs.index.astype(str)

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(
            season_avg_runs.index,
            season_avg_runs.values,
            marker="o",
            linewidth=2,
            color="#FF4B4B"
        )

        # Title
        # ax.set_title("Average Runs per Season", fontsize=12, fontweight="bold")

        # Axis labels
        ax.set_xlabel("Season", fontsize=10)
        ax.set_ylabel("Avg Runs per Match", fontsize=10)

        # Fix x-axis overlap
        ax.tick_params(axis='x', rotation=45)

        # Clean look (remove box)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Light grid only on Y-axis
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        plt.tight_layout()

        st.pyplot(fig)

    # ROW 2
    row2_col1, row2_col2 = st.columns(2)

    # Top Cities
    with row2_col1:
        st.markdown("Top 10 Cities by Total Runs")

        city_runs = df.groupby("city")["total_runs"].sum().sort_values(ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(6, 4))

        city_runs.sort_values().plot(kind="barh", ax=ax)

        # ax.set_title("Top 10 Cities by Total Runs")
        ax.set_xlabel("Total Runs Scored")
        ax.set_ylabel("City")

        st.pyplot(fig)

    # ⚖️ Toss Distribution
    with row2_col2:
        
        st.markdown("Toss Decision Distribution")

        toss_season = df.groupby(["season", "toss_decision"]).size().unstack().fillna(0)

        # -------------------------
        # PLOT
        # -------------------------
        fig, ax = plt.subplots(figsize=(8, 4))

        toss_season.plot(
            kind="bar",
            stacked=True,
            ax=ax
        )

        ax.set_xlabel("Season")
        ax.set_ylabel("Count")
        ax.legend(title="Decision")

        st.pyplot(fig)

    st.divider()

    # INSIGHTS SECTION


    st.markdown("Key Insights ")

    st.success("""
    - Matches increased from ~55 to ~75 (2007–2025)
    - Teams mostly choose to field first
    - Mumbai hosts the most matches 
    """)


with tab2:

    st.markdown("Batting Performance Analysis.")
    st.caption("Shows how players or teams perform while batting.")

    # FILTERED DATA 
    if selected_team != "All":
        batting_df = df[df["batting_team"] == selected_team].copy()
    else:
        batting_df = df.copy()

    # KPI 1: Highest Impact Batsman
    def calculate_phase_weight(row):
        if row['over'] <= 6:
            return 1.2
        elif row['over'] >= 16:
            return 1.5
        else:
            return 1.0

    batting_df['phase_weight'] = batting_df.apply(calculate_phase_weight, axis=1)
    batting_df['weighted_runs'] = batting_df['batsman_runs'] * batting_df['phase_weight']
    batting_df['is_dot'] = (batting_df['batsman_runs'] == 0) & (batting_df['valid_ball'] == 1)

    impact_stats = batting_df.groupby('batsman').agg({
        'weighted_runs': 'sum',
        'batsman_runs': 'sum',
        'valid_ball': 'sum',
        'is_dot': 'sum',
        'match_id': 'nunique'
    }).reset_index()

    impact_stats['strike_rate'] = (impact_stats['batsman_runs'] / impact_stats['valid_ball']) * 100
    impact_stats['impact_score'] = (
        impact_stats['weighted_runs'] +
        (impact_stats['strike_rate'] * 0.5) -
        (impact_stats['is_dot'] * 0.2)
    )

    impact_stats = impact_stats[impact_stats['match_id'] >= 10]
    top_impact = impact_stats.nlargest(1, 'impact_score').iloc[0]

    # KPI 2: Clutch King
    clutch_df = batting_df[batting_df['over'] >= 16]

    clutch_stats = clutch_df.groupby('batsman').agg({
        'batsman_runs': 'sum',
        'valid_ball': 'sum',
        'match_id': 'nunique'
    }).reset_index()

    clutch_stats['clutch_sr'] = (clutch_stats['batsman_runs'] / clutch_stats['valid_ball']) * 100
    clutch_stats = clutch_stats[
        (clutch_stats['match_id'] >= 5) &
        (clutch_stats['valid_ball'] >= 30)
    ]

    clutch_king = clutch_stats.nlargest(1, 'clutch_sr').iloc[0] if len(clutch_stats) > 0 else {'batsman': 'N/A', 'clutch_sr': 0}

    # KPI 3: Death Boundary King
    death_df = batting_df[batting_df['over'] >= 16].copy()
    death_df['is_boundary'] = death_df['batsman_runs'].isin([4, 6])

    boundary_stats = death_df.groupby('batsman').agg({
        'is_boundary': 'sum',
        'match_id': 'nunique'
    }).reset_index()

    boundary_stats = boundary_stats[boundary_stats['match_id'] >= 5]

    boundary_master = boundary_stats.nlargest(1, 'is_boundary').iloc[0] if len(boundary_stats) > 0 else {'batsman': 'N/A', 'is_boundary': 0}

    
    # KPI 4: Pressure Survivor
    
    pressure_df = batting_df[
        (batting_df['over'] <= 6) |
        (batting_df['over'] >= 16)
    ]

    survivor_stats = pressure_df.groupby('batsman').agg({
        'is_wicket': 'sum',
        'match_id': 'nunique',
        'valid_ball': 'sum'
    }).reset_index()

    survivor_stats['dismissal_rate'] = survivor_stats['is_wicket'] / survivor_stats['match_id']
    survivor_stats['survival_rate'] = (1 - survivor_stats['dismissal_rate']) * 100

    survivor_stats = survivor_stats[
        (survivor_stats['match_id'] >= 10) &
        (survivor_stats['valid_ball'] >= 50)
    ]

    survivor = survivor_stats.nlargest(1, 'survival_rate').iloc[0] if len(survivor_stats) > 0 else {'batsman': 'N/A', 'survival_rate': 0}

    # KPI DISPLAY (TOP ROW)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Highest Impact", top_impact['batsman'], f"{int(top_impact['impact_score'])} pts")

    with col2:
        st.metric("Clutch King", clutch_king['batsman'], f"SR: {clutch_king['clutch_sr']:.1f}")

    with col3:
        st.metric("Death Boundary King", boundary_master['batsman'], f"{int(boundary_master['is_boundary'])} boundaries")

    with col4:
        st.metric("Pressure Survivor", survivor['batsman'], f"{survivor['survival_rate']:.1f}%")

    st.markdown("---")

    # ROW 1 (2 COLUMNS)
    col1, col2 = st.columns(2)

    # CHART 1: TOP BATSMEN
    with col1:

        top_batsmen = (
            df.groupby("batsman")["batsman_runs"]
            .sum()
            .sort_values(ascending=True)
            .tail(10)
        )
        fig = px.bar(
        x=top_batsmen.values,
        y=top_batsmen.index,
        orientation='h',
        title="Top Batsmen"
        )

        fig.update_layout(
        xaxis_title="Runs",
        yaxis_title="Batsmen"
        )

        st.plotly_chart(fig, use_container_width=True)

    # CHART 2: BOUNDARY PATTERNS
    with col2:

        data = df.copy()
        data = data.sort_values(["match_id", "inning", "over_ball"])

        data["is_four"] = (data["batsman_runs"] == 4).astype(int)
        data["is_six"] = (data["batsman_runs"] == 6).astype(int)

        if selected_team == "All":

            agg = data.groupby("over_ball", as_index=False).agg({
                "is_four": "mean",
                "is_six": "mean"
            })

        else:

            agg = data.groupby("over_ball", as_index=False).agg({
                "is_four": "sum",
                "is_six": "sum"
            })

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=agg["over_ball"],
            y=agg["is_four"],
            mode="lines",
            name="4s"
        ))

        fig.add_trace(go.Scatter(
            x=agg["over_ball"],
            y=agg["is_six"],
            mode="lines",
            name="6s"
        ))

        fig.update_layout(
            title="Boundary Patterns",
            xaxis_title="Over Ball Progression",
            yaxis_title="Count",
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)

    # ROW 2 (2 COLUMNS)
    col3, col4 = st.columns(2)

    # CHART 3: EXTRAS
    with col3:

        extras_counts = df[[
            "isWide",
            "isNoBall",
            "Byes",
            "LegByes",
            "Penalty"
        ]].sum()

        fig = px.bar(
            x=extras_counts.index,
            y=extras_counts.values,
            text=extras_counts.values,
            title="Extras Distribution"
        )

        fig.update_layout(
            xaxis_title="Extras Type",
            yaxis_title="Count"
        )

        st.plotly_chart(fig, use_container_width=True)

    # CHART 4: RUN RATE
    with col4:

        over_runs = df.groupby("over")["total_runs"].sum().reset_index()
        over_runs["run_rate"] = over_runs["total_runs"] / 6

        fig = px.line(
            over_runs,
            x="over",
            y="run_rate",
            markers=True,
            title="Run Rate by Over"
        )

        fig.update_layout(
            xaxis_title="Over",
            yaxis_title="Run Rate"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

    # INSIGHTS SECTION


    st.markdown("Key Insights ")

    st.success("""
    - Boundaries (4s) dominate scoring  
    - Wides are the most common extras  
    - Run rate drops in early middle overs (5–6)  
    - Significant scoring boost from overs 6 to 17  
    """)
    st.info("""
    - Highest Impact Player shows overall match influence -> Batting + Match outcome
    - Clutch King shows performance in pressure situations -> Final Overs contribution
    - Death Boundary King shows scoring in final overs -> Number of 4s and 6s in death overs
    - Pressure Survivor shows stability in tough moments -> Low dismissal rate
    """)


with tab3:
    st.markdown("Bowling & Pressure")
    st.caption("Shows how bowlers perform and handle pressure situations")


    bowling_df = df.copy()

    # KPI ROW
  
    col1, col2, col3, col4 = st.columns(4)

    # Top Wicket Taker
    top_wickets = (
        bowling_df.groupby("bowler")["is_wicket"]
        .sum()
        .sort_values(ascending=False)
        .head(1)
    )

    # Best Economy
    eco_df = bowling_df.groupby("bowler").agg({
        "total_runs": "sum",
        "ball": "count"
    }).reset_index()

    eco_df["overs"] = eco_df["ball"] / 6
    eco_df["economy"] = eco_df["total_runs"] / eco_df["overs"]
    eco_df = eco_df[eco_df["overs"] >= 20].sort_values("economy")

    best_economy = eco_df.head(1)

    # Most Extras
    extras_df = bowling_df.groupby("bowler")[["isWide", "isNoBall"]].sum()
    extras_df["extras"] = extras_df["isWide"] + extras_df["isNoBall"]
    worst_extras = extras_df.sort_values("extras", ascending=False).head(1)

    # Death Overs Wickets
    death_df = bowling_df[bowling_df["over"] >= 16]
    death_wickets = (
        death_df.groupby("bowler")["is_wicket"]
        .sum()
        .sort_values(ascending=False)
        .head(1)
    )

    with col1:
        st.metric("Top Wicket Taker", top_wickets.index[0], int(top_wickets.values[0]))

    with col2:
        st.metric("Best Economy", best_economy["bowler"].values[0], f"{best_economy['economy'].values[0]:.2f}")

    with col3:
        st.metric("Most Extras", worst_extras.index[0], int(worst_extras["extras"].values[0]))

    with col4:
        st.metric("Death Overs Star", death_wickets.index[0], int(death_wickets.values[0]))

    st.markdown("---")

    # ROW 1
    col1, col2 = st.columns(2)

    # CHART 1: TOP WICKET TAKERS
    with col1:

        top_bowlers = (
            data.groupby("bowler")["is_wicket"]
            .sum()
            .sort_values(ascending=True)
            .tail(10)
        )

        fig = px.bar(
            x=top_bowlers.values,
            y=top_bowlers.index,
            orientation="h",
            title="Top Wicket Takers"
        )

        fig.update_layout(
            xaxis_title="Wickets",
            yaxis_title="Bowler"
        )

        st.plotly_chart(fig, use_container_width=True)

    # CHART 2: WICKETS BY PHASE
    with col2:

        data["phase"] = data["over"].apply(
            lambda x: "Powerplay" if x <= 6 else "Middle Overs" if x <= 15 else "Death Overs"
        )

        wicket_phase = (
            data[data["is_wicket"] == 1]
            .groupby("phase")["is_wicket"]
            .sum()
        )

        fig = px.bar(
            x=wicket_phase.index,
            y=wicket_phase.values,
            title="Wickets by Phase"
        )

        fig.update_layout(
            xaxis_title="Phase",
            yaxis_title="Wickets"
        )

        st.plotly_chart(fig, use_container_width=True)

    # ROW 2
    col3, col4 = st.columns(2)

    # CHART 3: WICKETS BY OVER
    with col3:

        wickets_by_over = (
            data[data["is_wicket"] == 1]
            .groupby("over")["is_wicket"]
            .sum()
            .reset_index()
        )

        fig = px.line(
            wickets_by_over,
            x="over",
            y="is_wicket",
            markers=True,
            title="Wickets by Over"
        )

        fig.update_layout(
            xaxis_title="Over",
            yaxis_title="Wickets"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col4:

        # Use already filtered dataset (season + team applied globally)
        data = bowling_df.copy()

        # Ensure dot balls
        data["is_dot"] = (data["batsman_runs"] == 0) & (data["valid_ball"] == 1)

        # Dot ball pressure by bowler
        dot_pressure = (
            data.groupby("bowler")
            .agg({
                "is_dot": "sum",
                "ball": "count"
            })
            .reset_index()
        )

        dot_pressure = dot_pressure[dot_pressure["ball"] >= 50]

        # Sort top pressure creators
        dot_pressure = dot_pressure.sort_values("is_dot", ascending=False).head(10)

        # Plot
        fig = px.bar(
            x=dot_pressure["is_dot"],
            y=dot_pressure["bowler"],
            orientation="h",
            title="Dot Ball Pressure Index"
        )

        fig.update_layout(
            xaxis_title="Dot Balls",
            yaxis_title="Bowler"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.markdown("Key Insights ")

    st.success("""
    - Wickets taken in Middle Overs Phase.
    - Wickets down in early Overs.
    """)

    st.info("""
    - Best Economy -> gives least runs
    - Most Extras -> gives extra free runs
    - Death Over Star -> performs best in final overs
    - Dot Balls -> No runs scored
            """)


with tab4:
    st.markdown(" Match Outcome Drivers")
    st.caption("What factors influence the match results.")

    data = df.copy()

    # ROW 1
    col1, col2 = st.columns(2)

    # CHART 1: TOSS DECISION IMPACT
    with col1:

        toss_impact = data.groupby("toss_decision")["match_id"].nunique()

        fig = px.bar(
            x=toss_impact.index,
            y=toss_impact.values,
            title="Toss Decision vs Match Wins"
        )

        fig.update_layout(
            xaxis_title="Toss Decision",
            yaxis_title="Matches Won"
        )

        st.plotly_chart(fig, use_container_width=True)


    # CHART 2: MATCH OUTCOME TYPE
    with col2:

        outcome_counts = data["outcome"].value_counts()

        fig = px.pie(
            names=outcome_counts.index,
            values=outcome_counts.values,
            title="Match Outcome Type Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    # ROW 2
    col3, col4 = st.columns(2)

    # CHART 3: TOSS WIN vs MATCH WIN
    with col3:

        toss_win_match = data.groupby("toss_win")["match_id"].nunique()
        toss_win_season = df.groupby(["season", "toss_win"])["match_id"].nunique().unstack()


        fig = px.bar(
            x=["Lost Toss", "Won Toss"],
            y=toss_win_match.values,
            title="Toss Win vs Match Win"
        )

        fig.update_layout(
            xaxis_title="Toss Result",
            yaxis_title="Matches Won"
        )

        st.plotly_chart(fig, use_container_width=True)


    st.markdown("Toss Win - Match Result (Team-wise)")
 
    match_df = df.drop_duplicates(subset="match_id")[
        ["match_id", "toss_winner", "winner"]
    ].copy()


    match_df["toss_result"] = (
        match_df["toss_winner"] == match_df["winner"]
    )

    match_df["toss_result"] = match_df["toss_result"].map({
        True: "Win",
        False: "Loss"
    })

    team_toss = (
        match_df.groupby(["toss_winner", "toss_result"])["match_id"]
        .count()
        .unstack()
        .fillna(0)
    )

    st.bar_chart(team_toss)


    
    # CHART 4: VENUE IMPACT
    with col4:

        venue_impact = data["venue"].value_counts().head(10)

        fig = px.bar(
            x=venue_impact.values,
            y=venue_impact.index,
            orientation="h",
            title="Top Venues by Matches"
        )

        fig.update_layout(
            xaxis_title="Matches",
            yaxis_title="Venue"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("Key Insights ")

    st.success("""
    - Teams mostly choose to field after winning toss  
    - Normal wins dominate over one-sided matches  
    - Matches are generally balanced and competitive         
    """)


with tab5:
    st.markdown("Clutch Moments ( Game Changers)")
    st.caption("Momentum shifts in matches: run rate changes, wicket collapses, scoring pressure, and game swings over overs.")
    
    data = df.copy()
    data = data.sort_values(["match_id", "over"])

    # FEATURE ENGINEERING

    data["wicket_flag"] = data["is_wicket"]

    # ROW 1
    col1, col2 = st.columns(2)

    
    # CHART 1: MOMENTUM SWING DETECTOR
    with col1:

        over_runs = (
            data.groupby(["match_id", "over"])["total_runs"]
            .sum()
            .reset_index()
        )

        over_runs["rolling_rr"] = (
            over_runs.groupby("match_id")["total_runs"]
            .rolling(3)
            .mean()
            .reset_index(level=0, drop=True)
        )

        swing = over_runs.groupby("over")["rolling_rr"].mean().reset_index()

        fig = px.line(
            swing,
            x="over",
            y="rolling_rr",
            markers=True,
            title="Momentum Swing Detector"
        )

        fig.update_layout(
            xaxis_title="Over",
            yaxis_title="Run Rate Change"
        )

        st.plotly_chart(fig, use_container_width=True)

    # CHART 2: COLLAPSE WINDOW DETECTOR
    with col2:

        collapse = (
            data.groupby(["match_id", "over"])["wicket_flag"]
            .sum()
            .reset_index()
        )

        collapse["collapse_window"] = (
            collapse.groupby("match_id")["wicket_flag"]
            .rolling(3)
            .sum()
            .reset_index(level=0, drop=True)
        )

        final_collapse = collapse.groupby("over")["collapse_window"].mean().reset_index()

        fig = px.bar(
            final_collapse,
            x="over",
            y="collapse_window",
            title="Collapse Windows (3-Over Wicket Bursts)"
        )

        fig.update_layout(
            xaxis_title="Over",
            yaxis_title="Avg Wickets in Collapse Window"
        )

        st.plotly_chart(fig, use_container_width=True)

    
    # ROW 2
    col3, col4 = st.columns(2)

    # CHART 3: MOMENTUM STABILITY MAP
    with col3:

        stability = data.groupby("over").agg({
            "total_runs": "mean",
            "is_wicket": "mean"
        }).reset_index()

        fig = px.scatter(
            stability,
            x="total_runs",
            y="is_wicket",
            size="over",
            title="Momentum Stability Map"
        )

        fig.update_layout(
            xaxis_title="Avg Runs per Over",
            yaxis_title="Avg Wickets per Over"
        )

        st.plotly_chart(fig, use_container_width=True)

    # CHART 4:  MOMENTUM INTENSITY CURVE
    with col4:

        intensity = data.groupby("over").agg({
            "total_runs": "sum",
            "is_wicket": "sum"
        }).reset_index()

        intensity["momentum_intensity"] = intensity["total_runs"] + (intensity["is_wicket"] * 10)

        fig = px.line(
            intensity,
            x="over",
            y="momentum_intensity",
            markers=True,
            title="Momentum Intensity Curve"
        )

        fig.update_layout(
            xaxis_title="Over",
            yaxis_title="Momentum Score"
        )

        st.plotly_chart(fig, use_container_width=True)
    st.markdown("Key match momentum indicators")
    st.info("""
        - Run rate changes show scoring speed shifts  
        - Wicket collapses highlight pressure phases  
        - Runs vs wickets shows scoring balance  
        - Momentum score shows match control over overs  
""")


with tab6:
    st.markdown("Teams")

    teams_list = sorted(
        pd.unique(df[["batting_team", "bowling_team"]].values.ravel("K"))
    )
    teams_list = [t for t in teams_list if pd.notna(t)]

    if selected_team != "All":
        teams_list = [selected_team]

    for team in teams_list:

        team_df = df[
            (df["batting_team"] == team) |
            (df["bowling_team"] == team)
        ]

        if team_df.empty:
            continue

        batting_players = team_df[team_df["batting_team"] == team]["batsman"].unique()
        bowling_players = team_df[team_df["bowling_team"] == team]["bowler"].unique()

        start_season = team_df["season"].min()
        end_season = team_df["season"].max()

        # wins = team_df["winner"].value_counts().get(team, 0)
        wins = team_df[team_df["winner"] == team]["match_id"].nunique()
        matches = team_df["match_id"].nunique()
        win_pct = round((wins / matches) * 100, 1) if matches > 0 else 0

        toss_wins = team_df[team_df["toss_winner"] == team]["match_id"].nunique()
        total_runs = team_df[team_df["batting_team"] == team]["total_runs"].sum()

        wickets = team_df[
            (team_df["bowling_team"] == team) &
            (team_df["is_wicket"] == 1)
        ]["match_id"].shape[0]


        with st.container():

            st.title(f"{team}")
            st.markdown(f"📅 Seasons Played: {start_season} → {end_season}")
            st.markdown(
                f"🎯 Matches: {matches}  |  🏆 Wins: {wins}  |  🪙 Toss Wins: {toss_wins}  |  📊 Win %: {win_pct}"
            )
            st.markdown(
                f"🏏 Runs: {total_runs:,}  |  🎳 Wickets: {wickets:,}"
            )
            #  Expand Teams
            with st.expander("🏏 Batting Players"):
                st.write(", ".join(batting_players) if len(batting_players) else "N/A")

            with st.expander("🎳 Bowling Players"):
                st.write(", ".join(bowling_players) if len(bowling_players) else "N/A")

            st.markdown("---")