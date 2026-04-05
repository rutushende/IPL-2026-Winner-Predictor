"""
=============================================================================
  IPL 2026 PREDICTION PROJECT
  Module: predict_2026.py  ← MAIN ENTRY POINT
  
  Run: python predict_2026.py

  This script:
    1. Generates historical IPL data (2008-2025)
    2. Engineers 23 ML features per match
    3. Trains 4 models (LR, RF, XGBoost, MLP)
    4. Evaluates & compares all models
    5. Builds 2026 team profiles
    6. Outputs final IPL 2026 predictions
    7. Saves all plots to outputs/
=============================================================================
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.makedirs("outputs", exist_ok=True)

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ── Project modules ──────────────────────────
from data.data_generator import (
    generate_match_data, compute_team_season_stats,
    compute_h2h, get_squads_df, SQUADS_2026, TEAMS, TITLES
)
from utils.feature_engineering import (
    build_match_features, build_2026_team_profile
)
from models.train_models import train_all_models, model_comparison_table, save_best_model
from utils.evaluation import (
    plot_confusion_matrix, plot_roc_curves, plot_model_comparison,
    plot_feature_importance, plot_win_probabilities,
    plot_team_radar, plot_historical_trends
)


# ══════════════════════════════════════════════
#  STEP 1 — DATA GENERATION
# ══════════════════════════════════════════════
print("\n" + "█"*60)
print("  IPL 2026 WINNER PREDICTION — AI/ML SYSTEM")
print("█"*60)

print("\n[STEP 1] Generating historical IPL data (2008–2025)...")
matches_df = generate_match_data()
stats_df = compute_team_season_stats(matches_df)
h2h_df = compute_h2h(matches_df)
squads_df = get_squads_df()
print(f"  ✔ Matches: {len(matches_df)} | Seasons: {matches_df['season'].nunique()}")
print(f"  ✔ Team-season records: {len(stats_df)}")
print(f"  ✔ Head-to-head pairs: {len(h2h_df)}")

# Save raw datasets
matches_df.to_csv("outputs/matches_2008_2025.csv", index=False)
stats_df.to_csv("outputs/team_season_stats.csv", index=False)
h2h_df.to_csv("outputs/head_to_head.csv", index=False)
squads_df.to_csv("outputs/squads_2026.csv", index=False)
print("  ✔ Raw datasets saved to outputs/")


# ══════════════════════════════════════════════
#  STEP 2 — FEATURE ENGINEERING
# ══════════════════════════════════════════════
print("\n[STEP 2] Engineering features...")
features_df = build_match_features(matches_df, stats_df, h2h_df, squads_df)
print(f"  ✔ Feature matrix: {features_df.shape[0]} rows × {features_df.shape[1]} columns")
print(f"  ✔ Feature list: {[c for c in features_df.columns if c not in ('team1','team2','season','venue','target')]}")
features_df.to_csv("outputs/feature_matrix.csv", index=False)


# ══════════════════════════════════════════════
#  STEP 3 — MODEL TRAINING
# ══════════════════════════════════════════════
print("\n[STEP 3] Training ML models...")
results, best_result, X_test, y_test = train_all_models(features_df)

comparison_df = model_comparison_table(results)
comparison_df.to_csv("outputs/model_comparison.csv", index=False)
print("\n  📊 Model Comparison:")
print(comparison_df.to_string(index=False))

save_best_model(best_result, "outputs/best_model.pkl")


# ══════════════════════════════════════════════
#  STEP 4 — EVALUATION PLOTS
# ══════════════════════════════════════════════
print("\n[STEP 4] Generating evaluation plots...")

plot_model_comparison(results, save_path="outputs/model_comparison.png")
plot_roc_curves(results, X_test, y_test, save_path="outputs/roc_curves.png")
plot_feature_importance(best_result, save_path="outputs/feature_importance.png")
plot_historical_trends(stats_df, save_path="outputs/historical_trends.png")

for r in results:
    safe_name = r["model_name"].replace(" ", "_").replace("(", "").replace(")", "")
    plot_confusion_matrix(
        r["confusion_matrix"], r["model_name"],
        save_path=f"outputs/cm_{safe_name}.png"
    )


# ══════════════════════════════════════════════
#  STEP 5 — 2026 PREDICTIONS
# ══════════════════════════════════════════════
print("\n[STEP 5] Building IPL 2026 predictions...")
team_profiles = build_2026_team_profile(stats_df, squads_df, matches_df, h2h_df)
team_profiles.to_csv("outputs/ipl2026_predictions.csv", index=False)

plot_win_probabilities(team_profiles, save_path="outputs/win_probabilities.png")
plot_team_radar(team_profiles, top_n=4, save_path="outputs/radar_top4.png")


# ══════════════════════════════════════════════
#  STEP 6 — FINAL REPORT
# ══════════════════════════════════════════════
print("\n" + "═"*60)
print("  IPL 2026 — FINAL PREDICTIONS")
print("═"*60)

# Top 4 (Playoffs)
top4 = team_profiles.head(4)
finalists = team_profiles.head(2)
winner = team_profiles.iloc[0]
dark_horse = team_profiles.iloc[4]  # 5th team

print("\n🏆 PLAYOFFS — TOP 4 TEAMS:")
print(f"{'Rank':<6} {'Team':<35} {'Win Probability':<18} {'Titles'}")
print("-"*65)
for rank, (_, row) in enumerate(top4.iterrows(), 1):
    medal = ["🥇", "🥈", "🥉", "4️⃣"][rank - 1]
    print(f"  {medal}  {row['team']:<33} {row['win_probability']:>6.2f}%"
          f"            {int(TITLES.get(row['team'], 0))}")

print(f"\n🏟️  FINALISTS:")
for _, row in finalists.iterrows():
    print(f"   ▶ {row['team']} ({row['win_probability']:.2f}%)")

print(f"\n🏆 IPL 2026 PREDICTED WINNER:")
print(f"   🎊 {winner['team'].upper()}")
print(f"   Win Probability : {winner['win_probability']:.2f}%")
print(f"   Batting Index   : {winner['batting_index']:.4f}")
print(f"   Bowling Index   : {winner['bowling_index']:.4f}")
print(f"   Experience Score: {winner['experience_score']:.4f}")
print(f"   Form Score      : {winner['form_score']:.4f}")
info = SQUADS_2026.get(winner["team"], {})
print(f"   Key Players     : {', '.join(info.get('key_players', []))}")
print(f"   Strengths       : {info.get('strengths', 'N/A')}")

print(f"\n🌟 DARK HORSE TEAM:")
print(f"   ⚡ {dark_horse['team']} ({dark_horse['win_probability']:.2f}%)")
dark_info = SQUADS_2026.get(dark_horse["team"], {})
print(f"   Why: {dark_info.get('strengths', 'Balanced squad with explosive potential')}")

print("\n📊 FULL TEAM RANKINGS:")
print(f"{'Rank':<6} {'Team':<35} {'Win %':<10} {'Strength':<12} {'Batting':<10} {'Bowling'}")
print("-"*80)
for i, (_, row) in enumerate(team_profiles.iterrows(), 1):
    print(f"  {i:<4} {row['team']:<35} {row['win_probability']:>5.2f}%"
          f"    {row['strength_score']:.4f}      {row['batting_index']:.4f}"
          f"    {row['bowling_index']:.4f}")


# ══════════════════════════════════════════════
#  STEP 7 — SQUAD DETAILS
# ══════════════════════════════════════════════
print("\n" + "═"*60)
print("  2026 SQUAD DETAILS — ALL TEAMS")
print("═"*60)
for team in TEAMS:
    data = SQUADS_2026.get(team, {})
    profile_row = team_profiles[team_profiles["team"] == team]
    prob = profile_row["win_probability"].values[0] if not profile_row.empty else 0
    print(f"\n{'─'*55}")
    print(f"  {team.upper()}")
    print(f"  Captain    : {data.get('captain', 'TBA')}")
    print(f"  Win Prob   : {prob:.2f}%")
    print(f"  Key Players: {', '.join(data.get('key_players', []))}")
    print(f"  Strengths  : {data.get('strengths', 'N/A')}")
    print(f"  Weaknesses : {data.get('weaknesses', 'N/A')}")
    squad = data.get("squad", [])
    playing_xi = data.get("playing_xi", [])
    print(f"  Full Squad ({len(squad)}): {', '.join(squad)}")
    print(f"  Playing XI: {', '.join(playing_xi)}")


print("\n" + "═"*60)
print("  ✅ ALL OUTPUTS SAVED TO /outputs/")
print("  📂 Files created:")
for f in sorted(os.listdir("outputs")):
    print(f"     • outputs/{f}")
print("═"*60)
print("\n  🚀 Run dashboard: streamlit run dashboard/app.py")
print("═"*60 + "\n")
