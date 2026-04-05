"""
=============================================================================
  IPL 2026 PREDICTION PROJECT
  Module: utils/feature_engineering.py
  Purpose: Build ML-ready feature vectors for each team matchup.

  FEATURES EXPLAINED
  ──────────────────
  1. team_strength_score
     = 0.4 * win_pct_last3yr + 0.3 * (titles/5) + 0.2 * nrr_proxy + 0.1 * form
     Captures overall team quality blending history and form.

  2. batting_strength_index
     = (batting_rating / 100) * (1 + 0.1 * avg_score_proxy)
     Higher means the team scores more runs consistently.

  3. bowling_strength_index
     = (bowling_rating / 100) * (1 + 0.1 * wicket_proxy)
     Higher means the team restricts opponents better.

  4. experience_score
     = (experience_rating / 100) * log(1 + titles)
     Logarithmic to prevent title-rich teams dominating.

  5. captaincy_factor
     = win_pct_as_captain (approximate from history)

  6. player_form_index
     = weighted average of last 10 match performances

  7. home_advantage_factor
     = 1.05 if playing at home ground, else 1.0

  8. head_to_head_win_rate
     = team_a_wins / total_h2h_matches

  9. toss_advantage
     = historical toss-to-win correlation for the team

  10. recent_form_score
      = wins in last 5 matches / 5
=============================================================================
"""

import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.data_generator import (
    TEAMS, SQUADS_2026, TITLES, generate_match_data,
    compute_team_season_stats, compute_h2h
)


# ─────────────────────────────────────────────
# INDIVIDUAL FEATURE CALCULATORS
# ─────────────────────────────────────────────

def compute_team_strength(stats_df: pd.DataFrame) -> pd.Series:
    """
    team_strength_score per team (most recent 3 seasons weighted).
    """
    recent = stats_df[stats_df["season"] >= 2022]
    grp = recent.groupby("team").agg(
        avg_win_pct=("win_pct", "mean"),
        total_wins=("wins", "sum"),
    ).reset_index()
    grp["title_factor"] = grp["team"].map(TITLES).fillna(0) / 5.0
    grp["strength"] = (
        0.50 * grp["avg_win_pct"]
        + 0.30 * grp["title_factor"]
        + 0.20 * (grp["total_wins"] / grp["total_wins"].max())
    )
    return grp.set_index("team")["strength"]


def compute_batting_index(squads_df: pd.DataFrame) -> pd.Series:
    """batting_strength_index = batting_rating / 100"""
    return (squads_df.set_index("team")["batting_rating"] / 100)


def compute_bowling_index(squads_df: pd.DataFrame) -> pd.Series:
    """bowling_strength_index = bowling_rating / 100"""
    return (squads_df.set_index("team")["bowling_rating"] / 100)


def compute_experience_score(squads_df: pd.DataFrame) -> pd.Series:
    """experience_score = (exp_rating/100) * log(1 + titles)"""
    sq = squads_df.set_index("team")
    return (sq["experience_rating"] / 100) * np.log1p(sq["titles"])


def compute_form(stats_df: pd.DataFrame, last_n: int = 5) -> pd.Series:
    """recent_form = win_pct in last N matches of most recent season"""
    last_season = stats_df["season"].max()
    recent = stats_df[stats_df["season"] >= last_season - 1]
    form = recent.groupby("team")["win_pct"].mean()
    return form


def compute_h2h_rate(h2h_df: pd.DataFrame) -> dict:
    """
    Returns dict: (team_a, team_b) -> team_a_win_rate
    """
    rates = {}
    for _, row in h2h_df.iterrows():
        total = max(row["total_matches"], 1)
        rates[(row["team_a"], row["team_b"])] = row["team_a_wins"] / total
        rates[(row["team_b"], row["team_a"])] = row["team_b_wins"] / total
    return rates


def home_advantage(team: str, venue: str) -> float:
    """1.05 boost if playing at home ground."""
    from data.data_generator import HOME_VENUES
    return 1.05 if HOME_VENUES.get(team, "") == venue else 1.0


def compute_toss_advantage(matches_df: pd.DataFrame) -> pd.Series:
    """
    toss_win_to_match_win correlation per team.
    toss_advantage = P(win | won toss)
    """
    results = {}
    for team in TEAMS:
        toss_won = matches_df[matches_df["toss_winner"] == team]
        won_after_toss = (toss_won["winner"] == team).sum()
        rate = won_after_toss / max(len(toss_won), 1)
        results[team] = rate
    return pd.Series(results)


# ─────────────────────────────────────────────
# BUILD FULL FEATURE MATRIX (for ML)
# ─────────────────────────────────────────────

def build_match_features(
    matches_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    h2h_df: pd.DataFrame,
    squads_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each historical match, build a feature vector:
    [team1_strength, team2_strength, team1_batting, team2_batting,
     team1_bowling, team2_bowling, team1_experience, team2_experience,
     team1_form, team2_form, h2h_rate, toss_advantage, home_adv,
     team_balance_diff] → target: 1 if team1 wins else 0
    """
    strength = compute_team_strength(stats_df)
    batting = compute_batting_index(squads_df)
    bowling = compute_bowling_index(squads_df)
    experience = compute_experience_score(squads_df)
    toss_adv = compute_toss_advantage(matches_df)
    form = compute_form(stats_df)
    h2h_rates = compute_h2h_rate(h2h_df)
    balance = squads_df.set_index("team")["team_balance"] / 100

    rows = []
    for _, m in matches_df.iterrows():
        t1, t2 = m["team1"], m["team2"]
        if t1 not in strength.index or t2 not in strength.index:
            continue

        row = {
            "team1": t1,
            "team2": t2,
            "season": m["season"],
            "venue": m["venue"],
            # Team 1 features
            "t1_strength": strength.get(t1, 0.5),
            "t1_batting": batting.get(t1, 0.85),
            "t1_bowling": bowling.get(t1, 0.82),
            "t1_experience": experience.get(t1, 0.5),
            "t1_form": form.get(t1, 0.5),
            "t1_toss_adv": toss_adv.get(t1, 0.5),
            "t1_balance": balance.get(t1, 0.85),
            "t1_titles": TITLES.get(t1, 0),
            # Team 2 features
            "t2_strength": strength.get(t2, 0.5),
            "t2_batting": batting.get(t2, 0.85),
            "t2_bowling": bowling.get(t2, 0.82),
            "t2_experience": experience.get(t2, 0.5),
            "t2_form": form.get(t2, 0.5),
            "t2_toss_adv": toss_adv.get(t2, 0.5),
            "t2_balance": balance.get(t2, 0.85),
            "t2_titles": TITLES.get(t2, 0),
            # Comparative features
            "h2h_t1_win_rate": h2h_rates.get((t1, t2), 0.5),
            "strength_diff": strength.get(t1, 0.5) - strength.get(t2, 0.5),
            "batting_diff": batting.get(t1, 0.85) - batting.get(t2, 0.85),
            "bowling_diff": bowling.get(t1, 0.82) - bowling.get(t2, 0.82),
            "form_diff": form.get(t1, 0.5) - form.get(t2, 0.5),
            "home_adv": 1.0 if m["home_team"] == t1 else 0.9,
            "toss_winner_is_t1": 1 if m["toss_winner"] == t1 else 0,
            # Target
            "target": 1 if m["winner"] == t1 else 0,
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 2026 PREDICTION FEATURE VECTOR
# ─────────────────────────────────────────────

def build_2026_team_profile(
    stats_df: pd.DataFrame,
    squads_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    h2h_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a 2026 feature profile for each team.
    Returns a DataFrame with one row per team.
    """
    strength = compute_team_strength(stats_df)
    batting = compute_batting_index(squads_df)
    bowling = compute_bowling_index(squads_df)
    experience = compute_experience_score(squads_df)
    form = compute_form(stats_df)
    toss_adv = compute_toss_advantage(matches_df)
    balance = squads_df.set_index("team")["team_balance"] / 100

    rows = []
    for team in TEAMS:
        rows.append({
            "team": team,
            "strength_score": round(strength.get(team, 0.5), 4),
            "batting_index": round(batting.get(team, 0.85), 4),
            "bowling_index": round(bowling.get(team, 0.82), 4),
            "experience_score": round(experience.get(team, 0.5), 4),
            "form_score": round(form.get(team, 0.5), 4),
            "toss_advantage": round(toss_adv.get(team, 0.5), 4),
            "team_balance": round(balance.get(team, 0.85), 4),
            "titles": TITLES.get(team, 0),
        })

    df = pd.DataFrame(rows)
    # Composite Score (normalized)
    df["composite_score"] = (
        0.25 * df["strength_score"]
        + 0.20 * df["batting_index"]
        + 0.20 * df["bowling_index"]
        + 0.15 * df["experience_score"]
        + 0.10 * df["form_score"]
        + 0.10 * df["team_balance"]
    )
    total = df["composite_score"].sum()
    df["win_probability"] = (df["composite_score"] / total * 100).round(2)
    return df.sort_values("win_probability", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from data.data_generator import compute_team_season_stats, compute_h2h, get_squads_df

    print("Generating data...")
    matches = generate_match_data()
    stats = compute_team_season_stats(matches)
    h2h = compute_h2h(matches)
    squads = get_squads_df()

    print("\n📊 2026 Team Profiles:")
    profile = build_2026_team_profile(stats, squads, matches, h2h)
    print(profile[["team", "strength_score", "batting_index",
                   "bowling_index", "win_probability"]].to_string())

    print("\n🔧 Building match feature matrix...")
    features = build_match_features(matches, stats, h2h, squads)
    print(f"  Feature matrix shape: {features.shape}")
    print(f"  Features: {[c for c in features.columns if c not in ('team1','team2','season','venue','target')]}")
