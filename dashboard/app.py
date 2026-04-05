"""
=============================================================================
  IPL 2026 PREDICTION PROJECT
  Module: dashboard/app.py
  Purpose: Streamlit web dashboard for IPL 2026 predictions.

  Run: streamlit run dashboard/app.py
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.data_generator import (
    generate_match_data, compute_team_season_stats,
    compute_h2h, get_squads_df, SQUADS_2026, TEAMS, TITLES
)
from utils.feature_engineering import build_2026_team_profile, build_match_features
from models.train_models import train_all_models, model_comparison_table

# ── Streamlit Config ─────────────────────────
st.set_page_config(
    page_title="IPL 2026 Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

TEAM_COLORS = {
    "Mumbai Indians": "#004BA0",
    "Chennai Super Kings": "#FFCC00",
    "Royal Challengers Bengaluru": "#E02020",
    "Kolkata Knight Riders": "#3A225D",
    "Rajasthan Royals": "#E91E8C",
    "Sunrisers Hyderabad": "#F26522",
    "Delhi Capitals": "#0078BC",
    "Punjab Kings": "#ED1B24",
    "Gujarat Titans": "#1C2B57",
    "Lucknow Super Giants": "#A2DFFF",
}

# ── Load / Cache Data ─────────────────────────
@st.cache_data
def load_all_data():
    matches = generate_match_data()
    stats = compute_team_season_stats(matches)
    h2h = compute_h2h(matches)
    squads = get_squads_df()
    team_profiles = build_2026_team_profile(stats, squads, matches, h2h)
    features = build_match_features(matches, stats, h2h, squads)
    return matches, stats, h2h, squads, team_profiles, features


@st.cache_resource
def train_models_cached(features):
    results, best, X_test, y_test = train_all_models(features)
    return results, best, X_test, y_test


# ══════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/5/5b/Indian_Premier_League_Logo.svg",
             width=130)
    st.markdown("## 🏏 IPL 2026 Predictor")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏆 Final Prediction",
        "📊 Team Analysis",
        "🤖 Model Comparison",
        "📈 Historical Trends",
        "🗂️ Squad Viewer",
        "⚔️ Head-to-Head",
    ])
    st.markdown("---")
    st.markdown("**Model:** XGBoost / Random Forest")
    st.markdown("**Data:** IPL 2008–2025")
    st.markdown("**Features:** 23 engineered")


# ── Load Data ────────────────────────────────
with st.spinner("Loading IPL data and training models..."):
    matches, stats, h2h, squads, team_profiles, features = load_all_data()
    results, best, X_test, y_test = train_models_cached(features)


# ══════════════════════════════════════════════
#  PAGE: FINAL PREDICTION
# ══════════════════════════════════════════════
if "Final Prediction" in page:
    st.title("🏆 IPL 2026 — Final Prediction")
    st.markdown("*Powered by Machine Learning | Historical data 2008–2025*")
    st.divider()

    winner = team_profiles.iloc[0]
    runner_up = team_profiles.iloc[1]
    dark_horse = team_profiles.iloc[4]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🥇 Predicted Winner", winner["team"],
                  f"{winner['win_probability']:.1f}% probability")
    with col2:
        st.metric("🥈 Runner-Up", runner_up["team"],
                  f"{runner_up['win_probability']:.1f}% probability")
    with col3:
        st.metric("⚡ Dark Horse", dark_horse["team"],
                  f"{dark_horse['win_probability']:.1f}% probability")

    st.divider()
    st.subheader("📊 All Teams — Win Probability")

    fig, ax = plt.subplots(figsize=(10, 5))
    df_plot = team_profiles.sort_values("win_probability", ascending=True)
    colors = [TEAM_COLORS.get(t, "#888") for t in df_plot["team"]]
    bars = ax.barh(df_plot["team"], df_plot["win_probability"], color=colors, edgecolor="white")
    for bar, val in zip(bars, df_plot["win_probability"]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left", fontsize=10, fontweight="bold")
    ax.set_xlabel("Win Probability (%)")
    ax.set_title("IPL 2026 Win Probability by Team", fontsize=14, fontweight="bold")
    ax.set_xlim(0, df_plot["win_probability"].max() + 4)
    ax.grid(axis="x", alpha=0.3)
    st.pyplot(fig)
    plt.close()

    st.divider()
    st.subheader("🏟️ Playoff Bracket")
    col1, col2, col3 = st.columns(3)
    top4 = team_profiles.head(4)
    with col1:
        st.markdown("**Qualifier 1**")
        st.info(f"🔵 {top4.iloc[0]['team']} ({top4.iloc[0]['win_probability']:.1f}%)")
        st.info(f"🟡 {top4.iloc[1]['team']} ({top4.iloc[1]['win_probability']:.1f}%)")
    with col2:
        st.markdown("**Eliminator**")
        st.warning(f"🟠 {top4.iloc[2]['team']} ({top4.iloc[2]['win_probability']:.1f}%)")
        st.warning(f"🔴 {top4.iloc[3]['team']} ({top4.iloc[3]['win_probability']:.1f}%)")
    with col3:
        st.markdown("**Final**")
        st.success(f"🏆 **{winner['team']}** WINS IPL 2026!")


# ══════════════════════════════════════════════
#  PAGE: TEAM ANALYSIS
# ══════════════════════════════════════════════
elif "Team Analysis" in page:
    st.title("📊 Team Analysis")
    team_sel = st.selectbox("Select Team", TEAMS)

    profile = team_profiles[team_profiles["team"] == team_sel].iloc[0]
    squad_data = SQUADS_2026.get(team_sel, {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Win Probability", f"{profile['win_probability']:.1f}%")
    col2.metric("Batting Index", f"{profile['batting_index']:.3f}")
    col3.metric("Bowling Index", f"{profile['bowling_index']:.3f}")
    col4.metric("Titles", TITLES.get(team_sel, 0))

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏏 Full Squad")
        squad = squad_data.get("squad", [])
        xi = squad_data.get("playing_xi", [])
        for p in squad:
            badge = "✅" if p in xi else "🔵"
            st.write(f"{badge} {p}")
        st.caption("✅ = Playing XI | 🔵 = Squad member")

    with col2:
        st.subheader("📋 Analysis")
        st.write(f"**Captain:** {squad_data.get('captain', 'N/A')}")
        st.write(f"**Key Players:** {', '.join(squad_data.get('key_players', []))}")
        st.success(f"**Strengths:** {squad_data.get('strengths', 'N/A')}")
        st.error(f"**Weaknesses:** {squad_data.get('weaknesses', 'N/A')}")

        # Radar
        categories = ["Batting", "Bowling", "Experience", "Form", "Balance"]
        ratings = squad_data
        vals = [
            ratings.get("batting_rating", 80) / 100,
            ratings.get("bowling_rating", 80) / 100,
            ratings.get("experience_rating", 80) / 100,
            float(profile["form_score"]),
            ratings.get("team_balance", 80) / 100,
        ]

        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        vals_plot = vals + vals[:1]

        fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=9)
        ax.plot(angles, vals_plot, "o-", linewidth=2,
                color=TEAM_COLORS.get(team_sel, "#333"))
        ax.fill(angles, vals_plot, alpha=0.25,
                color=TEAM_COLORS.get(team_sel, "#333"))
        ax.set_ylim(0, 1)
        ax.set_title(team_sel, fontsize=10, fontweight="bold", pad=20)
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════
#  PAGE: MODEL COMPARISON
# ══════════════════════════════════════════════
elif "Model Comparison" in page:
    st.title("🤖 ML Model Comparison")

    comp_df = model_comparison_table(results)
    st.dataframe(comp_df.style.highlight_max(
        subset=["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"],
        color="#d4edda"
    ), use_container_width=True)

    st.divider()
    st.subheader("📈 Performance Chart")

    metrics = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
    x = np.arange(len(metrics))
    width = 0.8 / len(results)
    colors = ["#FF6B35", "#004E89", "#1A936F", "#C62828"]

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, r in enumerate(results):
        vals = [r[m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=r["model_name"],
               color=colors[i % len(colors)], alpha=0.85, edgecolor="white")

    ax.set_xticks(x + width * (len(results) - 1) / 2)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)
    plt.close()

    st.divider()
    st.subheader("🔍 Best Model Confusion Matrix")
    cm = best["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Team2 Wins", "Team1 Wins"],
                yticklabels=["Team2 Wins", "Team1 Wins"], ax=ax)
    ax.set_title(f"Confusion Matrix — {best['model_name']}", fontweight="bold")
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════
#  PAGE: HISTORICAL TRENDS
# ══════════════════════════════════════════════
elif "Historical" in page:
    st.title("📈 Historical Win Trends (2008–2025)")

    teams_sel = st.multiselect("Select Teams", TEAMS, default=TEAMS[:5])
    if teams_sel:
        fig, ax = plt.subplots(figsize=(12, 5))
        colors_list = list(TEAM_COLORS.values())
        for i, team in enumerate(teams_sel):
            td = stats[stats["team"] == team].sort_values("season")
            ax.plot(td["season"], td["win_pct"], marker="o", linewidth=2,
                    label=team, color=colors_list[i % len(colors_list)])
        ax.set_xlabel("Season")
        ax.set_ylabel("Win Percentage")
        ax.set_title("Win Percentage Trends")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

    st.divider()
    st.subheader("Season-wise Stats")
    season_sel = st.slider("Select Season", 2008, 2025, 2024)
    st.dataframe(stats[stats["season"] == season_sel][
        ["team", "matches_played", "wins", "losses", "win_pct"]
    ].sort_values("win_pct", ascending=False), use_container_width=True)


# ══════════════════════════════════════════════
#  PAGE: SQUAD VIEWER
# ══════════════════════════════════════════════
elif "Squad" in page:
    st.title("🗂️ 2026 Squad Viewer")
    st.dataframe(
        squads[["team", "captain", "batting_rating", "bowling_rating",
                "team_balance", "experience_rating", "titles"]]
        .sort_values("batting_rating", ascending=False),
        use_container_width=True
    )
    st.divider()
    for team in TEAMS:
        with st.expander(f"🏏 {team}"):
            data = SQUADS_2026[team]
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**Squad:** {', '.join(data['squad'])}")
            with c2:
                st.write(f"**Playing XI:** {', '.join(data['playing_xi'])}")
                st.write(f"**Captain:** {data['captain']}")
                st.write(f"**Key:** {', '.join(data['key_players'])}")


# ══════════════════════════════════════════════
#  PAGE: HEAD-TO-HEAD
# ══════════════════════════════════════════════
elif "Head-to-Head" in page:
    st.title("⚔️ Head-to-Head Analysis")
    col1, col2 = st.columns(2)
    with col1:
        t1 = st.selectbox("Team A", TEAMS, index=0)
    with col2:
        t2 = st.selectbox("Team B", TEAMS, index=1)

    if t1 != t2:
        mask = ((h2h["team_a"] == t1) & (h2h["team_b"] == t2)) | \
               ((h2h["team_a"] == t2) & (h2h["team_b"] == t1))
        row = h2h[mask]
        if not row.empty:
            r = row.iloc[0]
            t1_wins = r["team_a_wins"] if r["team_a"] == t1 else r["team_b_wins"]
            t2_wins = r["team_b_wins"] if r["team_a"] == t1 else r["team_a_wins"]
            total = r["total_matches"]

            col1, col2, col3 = st.columns(3)
            col1.metric(f"{t1} Wins", t1_wins)
            col2.metric("Total Matches", total)
            col3.metric(f"{t2} Wins", t2_wins)

            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar([t1, t2], [t1_wins, t2_wins],
                   color=[TEAM_COLORS.get(t1, "#333"), TEAM_COLORS.get(t2, "#888")])
            ax.set_title(f"H2H: {t1} vs {t2}", fontweight="bold")
            ax.set_ylabel("Wins")
            st.pyplot(fig)
            plt.close()


# ── Footer ────────────────────────────────────
st.divider()
st.caption("🏏 IPL 2026 Prediction System | Built with Python, Scikit-learn & Streamlit | For educational purposes")
