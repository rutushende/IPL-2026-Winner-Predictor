"""
=============================================================================
  IPL 2026 PREDICTION PROJECT
  Module: notebooks/ipl_analysis.py
  Purpose: Exploratory Data Analysis — run cell by cell like a notebook.
           Covers descriptive stats, visualizations, correlation analysis,
           and feature distribution checks.
  
  How to use:
    python notebooks/ipl_analysis.py
  
  Or paste sections into Jupyter for interactive exploration.
=============================================================================
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.makedirs("outputs", exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from data.data_generator import (
    generate_match_data, compute_team_season_stats,
    compute_h2h, get_squads_df, TEAMS, TITLES, SQUADS_2026
)
from utils.feature_engineering import (
    build_match_features, build_2026_team_profile
)

sns.set_theme(style="whitegrid")
PALETTE = sns.color_palette("tab10", 10)

print("=" * 65)
print("  IPL 2026 — EXPLORATORY DATA ANALYSIS")
print("=" * 65)


# ══════════════════════════════════════════════
#  CELL 1: Load Data
# ══════════════════════════════════════════════
print("\n[Cell 1] Loading data...")
matches = generate_match_data()
stats = compute_team_season_stats(matches)
h2h = compute_h2h(matches)
squads = get_squads_df()
features = build_match_features(matches, stats, h2h, squads)
profiles = build_2026_team_profile(stats, squads, matches, h2h)

print(matches.head(3).to_string())
print(f"\nShape: {matches.shape}")
print(f"Seasons: {sorted(matches['season'].unique())}")


# ══════════════════════════════════════════════
#  CELL 2: Basic Stats
# ══════════════════════════════════════════════
print("\n[Cell 2] Basic match statistics...")
print(f"Total matches: {len(matches)}")
print(f"Total seasons: {matches['season'].nunique()}")
print(f"Unique venues: {matches['venue'].nunique()}")
print(f"\nMatches per season:\n{matches['season'].value_counts().sort_index()}")

# Wins per team (all time)
wins = matches.groupby("winner").size().sort_values(ascending=False)
print(f"\nAll-time wins:\n{wins}")


# ══════════════════════════════════════════════
#  CELL 3: Win Rate Analysis
# ══════════════════════════════════════════════
print("\n[Cell 3] Win rate analysis...")
overall_stats = stats.groupby("team").agg(
    total_matches=("matches_played", "sum"),
    total_wins=("wins", "sum"),
).reset_index()
overall_stats["overall_win_pct"] = (
    overall_stats["total_wins"] / overall_stats["total_matches"]
).round(4)
overall_stats = overall_stats.sort_values("overall_win_pct", ascending=False)
print(overall_stats.to_string(index=False))


# ══════════════════════════════════════════════
#  CELL 4: Toss Analysis
# ══════════════════════════════════════════════
print("\n[Cell 4] Toss analysis...")
toss_win = matches[matches["toss_winner"] == matches["winner"]]
print(f"Toss winner wins the match: {len(toss_win)/len(matches)*100:.1f}%")

toss_dec = matches["toss_decision"].value_counts()
print(f"\nToss decisions:\n{toss_dec}")


# ══════════════════════════════════════════════
#  CELL 5: Score Distribution
# ══════════════════════════════════════════════
print("\n[Cell 5] Score distribution...")
print(matches[["team1_score", "team2_score"]].describe().round(2))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(matches["team1_score"], bins=30, color="#FF6B35", alpha=0.8, edgecolor="white")
axes[0].set_title("Score Distribution (Batting First)", fontweight="bold")
axes[0].set_xlabel("Runs Scored")
axes[1].hist(matches["team1_score"] - matches["team2_score"], bins=30,
             color="#004E89", alpha=0.8, edgecolor="white")
axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
axes[1].set_title("Score Margin Distribution", fontweight="bold")
axes[1].set_xlabel("Margin (+ = Team1 Won)")
plt.tight_layout()
plt.savefig("outputs/eda_score_distribution.png", dpi=150)
plt.close()
print("  Saved: outputs/eda_score_distribution.png")


# ══════════════════════════════════════════════
#  CELL 6: Win % Heatmap (Team vs Season)
# ══════════════════════════════════════════════
print("\n[Cell 6] Win% heatmap...")
pivot = stats.pivot_table(values="win_pct", index="team", columns="season", aggfunc="mean")
fig, ax = plt.subplots(figsize=(16, 6))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn",
            linewidths=0.5, linecolor="white", ax=ax,
            vmin=0, vmax=1, annot_kws={"size": 7})
ax.set_title("Win Percentage Heatmap — All Teams Across Seasons",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Season")
ax.set_ylabel("Team")
plt.tight_layout()
plt.savefig("outputs/eda_heatmap.png", dpi=150)
plt.close()
print("  Saved: outputs/eda_heatmap.png")


# ══════════════════════════════════════════════
#  CELL 7: Feature Correlation Matrix
# ══════════════════════════════════════════════
print("\n[Cell 7] Feature correlation matrix...")
numeric_features = [
    "t1_strength", "t1_batting", "t1_bowling", "t1_experience",
    "t1_form", "strength_diff", "batting_diff", "bowling_diff",
    "form_diff", "h2h_t1_win_rate", "home_adv", "target"
]
corr = features[numeric_features].corr()
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            mask=mask, ax=ax, vmin=-1, vmax=1,
            linewidths=0.5, annot_kws={"size": 8})
ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("outputs/eda_correlation.png", dpi=150)
plt.close()
print("  Saved: outputs/eda_correlation.png")


# ══════════════════════════════════════════════
#  CELL 8: 2026 Team Profile Summary
# ══════════════════════════════════════════════
print("\n[Cell 8] 2026 team profiles...")
print(profiles[["team", "batting_index", "bowling_index",
               "experience_score", "form_score",
               "composite_score", "win_probability"]].to_string(index=False))


# ══════════════════════════════════════════════
#  CELL 9: H2H Heatmap
# ══════════════════════════════════════════════
print("\n[Cell 9] Head-to-head win rate heatmap...")
teams_list = TEAMS
h2h_matrix = pd.DataFrame(0.5, index=teams_list, columns=teams_list)

for _, row in h2h.iterrows():
    ta, tb = row["team_a"], row["team_b"]
    total = max(row["total_matches"], 1)
    if ta in teams_list and tb in teams_list:
        h2h_matrix.loc[ta, tb] = row["team_a_wins"] / total
        h2h_matrix.loc[tb, ta] = row["team_b_wins"] / total

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.eye(len(teams_list), dtype=bool)
sns.heatmap(h2h_matrix, annot=True, fmt=".2f", cmap="RdYlGn",
            mask=mask, ax=ax, vmin=0, vmax=1,
            linewidths=0.5, linecolor="white", annot_kws={"size": 8})
ax.set_title("Head-to-Head Win Rate Matrix\n(Row team vs Column team)",
             fontsize=13, fontweight="bold", pad=12)
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig("outputs/eda_h2h_heatmap.png", dpi=150)
plt.close()
print("  Saved: outputs/eda_h2h_heatmap.png")


# ══════════════════════════════════════════════
#  CELL 10: Squad Ratings Comparison
# ══════════════════════════════════════════════
print("\n[Cell 10] Squad ratings comparison...")
sq_plot = squads.sort_values("batting_rating", ascending=False)

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(sq_plot))
width = 0.25
ax.bar(x - width, sq_plot["batting_rating"], width, label="Batting Rating",
       color="#FF6B35", alpha=0.85)
ax.bar(x, sq_plot["bowling_rating"], width, label="Bowling Rating",
       color="#004E89", alpha=0.85)
ax.bar(x + width, sq_plot["team_balance"], width, label="Team Balance",
       color="#1A936F", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(sq_plot["team"], rotation=35, ha="right", fontsize=8)
ax.set_ylabel("Rating")
ax.set_title("2026 Squad Ratings Comparison", fontsize=14, fontweight="bold")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/eda_squad_ratings.png", dpi=150)
plt.close()
print("  Saved: outputs/eda_squad_ratings.png")

print("\n✅ EDA complete! All charts saved to outputs/")
print("   Files: eda_score_distribution, eda_heatmap, eda_correlation,")
print("          eda_h2h_heatmap, eda_squad_ratings")
