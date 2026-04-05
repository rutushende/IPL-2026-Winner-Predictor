"""
=============================================================================
  IPL 2026 PREDICTION PROJECT
  Module: data/data_generator.py
  Purpose: Generate realistic synthetic IPL match data (2008–2025)
           and 2026 squad data for all 10 teams.
=============================================================================
"""

import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
# TEAM DEFINITIONS
# ─────────────────────────────────────────────
TEAMS = [
    "Mumbai Indians",
    "Chennai Super Kings",
    "Royal Challengers Bengaluru",
    "Kolkata Knight Riders",
    "Rajasthan Royals",
    "Sunrisers Hyderabad",
    "Delhi Capitals",
    "Punjab Kings",
    "Gujarat Titans",
    "Lucknow Super Giants",
]

# Historical titles (approximate)
TITLES = {
    "Mumbai Indians": 5,
    "Chennai Super Kings": 5,
    "Kolkata Knight Riders": 3,
    "Rajasthan Royals": 2,
    "Sunrisers Hyderabad": 1,
    "Gujarat Titans": 1,
    "Royal Challengers Bengaluru": 0,
    "Delhi Capitals": 0,
    "Punjab Kings": 0,
    "Lucknow Super Giants": 0,
}

# Base win probability (reflects historical dominance)
TEAM_BASE_WIN_PROB = {
    "Mumbai Indians": 0.58,
    "Chennai Super Kings": 0.57,
    "Kolkata Knight Riders": 0.52,
    "Rajasthan Royals": 0.50,
    "Royal Challengers Bengaluru": 0.48,
    "Sunrisers Hyderabad": 0.48,
    "Delhi Capitals": 0.46,
    "Gujarat Titans": 0.52,
    "Lucknow Super Giants": 0.48,
    "Punjab Kings": 0.44,
}

VENUES = [
    "Wankhede Stadium, Mumbai",
    "M. A. Chidambaram Stadium, Chennai",
    "Eden Gardens, Kolkata",
    "M. Chinnaswamy Stadium, Bengaluru",
    "Sawai Mansingh Stadium, Jaipur",
    "Rajiv Gandhi Intl. Stadium, Hyderabad",
    "Arun Jaitley Stadium, Delhi",
    "Punjab Cricket Association Stadium, Mohali",
    "Narendra Modi Stadium, Ahmedabad",
    "BRSABV Ekana Cricket Stadium, Lucknow",
    "Dr. DY Patil Sports Academy, Mumbai",
    "Brabourne Stadium, Mumbai",
]

HOME_VENUES = {
    "Mumbai Indians": "Wankhede Stadium, Mumbai",
    "Chennai Super Kings": "M. A. Chidambaram Stadium, Chennai",
    "Kolkata Knight Riders": "Eden Gardens, Kolkata",
    "Royal Challengers Bengaluru": "M. Chinnaswamy Stadium, Bengaluru",
    "Rajasthan Royals": "Sawai Mansingh Stadium, Jaipur",
    "Sunrisers Hyderabad": "Rajiv Gandhi Intl. Stadium, Hyderabad",
    "Delhi Capitals": "Arun Jaitley Stadium, Delhi",
    "Punjab Kings": "Punjab Cricket Association Stadium, Mohali",
    "Gujarat Titans": "Narendra Modi Stadium, Ahmedabad",
    "Lucknow Super Giants": "BRSABV Ekana Cricket Stadium, Lucknow",
}

# Teams that existed in each season (GT and LSG started 2022)
SEASON_TEAMS = {
    year: (TEAMS if year >= 2022 else [t for t in TEAMS
           if t not in ("Gujarat Titans", "Lucknow Super Giants")])
    for year in range(2008, 2026)
}

# ─────────────────────────────────────────────
# GENERATE MATCH DATASET
# ─────────────────────────────────────────────
def generate_match_data() -> pd.DataFrame:
    """
    Simulate IPL match results from 2008 to 2025.
    Returns a DataFrame with one row per match.
    """
    matches = []
    match_id = 1

    for year in range(2008, 2026):
        teams = SEASON_TEAMS[year]
        n_teams = len(teams)
        # Round-robin (each pair plays twice) + playoffs (approx)
        matchups = [(t1, t2) for i, t1 in enumerate(teams)
                    for t2 in teams[i + 1:]]
        matchups = matchups * 2  # home + away

        for team1, team2 in matchups:
            venue = HOME_VENUES.get(team1, random.choice(VENUES))
            toss_winner = random.choice([team1, team2])
            toss_decision = random.choice(["bat", "field"])

            # Batting first team
            batting_first = team1 if toss_decision == "bat" and toss_winner == team1 else \
                            team2 if toss_decision == "bat" and toss_winner == team2 else \
                            team2 if toss_winner == team1 else team1
            bowling_first = team2 if batting_first == team1 else team1

            # Score simulation
            bat_strength = TEAM_BASE_WIN_PROB.get(batting_first, 0.5)
            bowl_strength = 1 - TEAM_BASE_WIN_PROB.get(bowling_first, 0.5)
            score1 = int(np.random.normal(165 + bat_strength * 15, 20))
            score1 = max(80, min(score1, 260))
            score2 = int(np.random.normal(158 + bowl_strength * 10, 22))
            score2 = max(70, min(score2, 250))

            winner = batting_first if score1 > score2 else bowling_first
            margin = abs(score1 - score2)

            matches.append({
                "match_id": match_id,
                "season": year,
                "team1": team1,
                "team2": team2,
                "venue": venue,
                "toss_winner": toss_winner,
                "toss_decision": toss_decision,
                "batting_first": batting_first,
                "bowling_first": bowling_first,
                "team1_score": score1 if batting_first == team1 else score2,
                "team2_score": score2 if batting_first == team1 else score1,
                "winner": winner,
                "win_margin_runs": margin if batting_first == winner else 0,
                "win_margin_wickets": random.randint(1, 9) if batting_first != winner else 0,
                "is_playoff": 0,
                "home_team": team1,
            })
            match_id += 1

    df = pd.DataFrame(matches)
    return df


# ─────────────────────────────────────────────
# TEAM SEASON STATISTICS
# ─────────────────────────────────────────────
def compute_team_season_stats(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute season-wise win/loss/NRR stats for each team.
    """
    rows = []
    for season in matches_df["season"].unique():
        sdf = matches_df[matches_df["season"] == season]
        for team in SEASON_TEAMS[season]:
            played = sdf[(sdf["team1"] == team) | (sdf["team2"] == team)]
            wins = (played["winner"] == team).sum()
            losses = len(played) - wins
            win_pct = wins / len(played) if len(played) > 0 else 0

            avg_score_batting = played[played["batting_first"] == team]["team1_score"].mean()
            avg_score_bowling = played[played["bowling_first"] == team]["team2_score"].mean()

            rows.append({
                "season": season,
                "team": team,
                "matches_played": len(played),
                "wins": wins,
                "losses": losses,
                "win_pct": round(win_pct, 4),
                "titles": TITLES.get(team, 0),
                "avg_batting_score": round(avg_score_batting or 155, 2),
                "avg_bowling_conceded": round(avg_score_bowling or 158, 2),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# HEAD-TO-HEAD STATS
# ─────────────────────────────────────────────
def compute_h2h(matches_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    teams = TEAMS
    for i, t1 in enumerate(teams):
        for t2 in teams[i + 1:]:
            h2h = matches_df[
                ((matches_df["team1"] == t1) & (matches_df["team2"] == t2)) |
                ((matches_df["team1"] == t2) & (matches_df["team2"] == t1))
            ]
            t1_wins = (h2h["winner"] == t1).sum()
            t2_wins = (h2h["winner"] == t2).sum()
            rows.append({
                "team_a": t1, "team_b": t2,
                "total_matches": len(h2h),
                "team_a_wins": t1_wins,
                "team_b_wins": t2_wins,
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 2026 SQUADS & PLAYING XI
# ─────────────────────────────────────────────
SQUADS_2026 = {
    "Mumbai Indians": {
        "squad": [
            "Rohit Sharma", "Ishan Kishan", "Suryakumar Yadav", "Tilak Varma",
            "Hardik Pandya", "Tim David", "Naman Dhir", "Kieron Pollard",
            "Jasprit Bumrah", "Trent Boult", "Piyush Chawla", "Mohammed Siraj",
            "Dewald Brevis", "Shams Mulani", "Nuwan Thushara",
        ],
        "playing_xi": [
            "Rohit Sharma", "Ishan Kishan", "Suryakumar Yadav", "Tilak Varma",
            "Hardik Pandya", "Tim David", "Naman Dhir", "Kieron Pollard",
            "Jasprit Bumrah", "Trent Boult", "Piyush Chawla",
        ],
        "captain": "Hardik Pandya",
        "key_players": ["Jasprit Bumrah", "Suryakumar Yadav", "Rohit Sharma"],
        "strengths": "World-class pace attack, explosive batting depth, experienced squad",
        "weaknesses": "Middle-order inconsistency, spinner dependency",
        "batting_rating": 88,
        "bowling_rating": 92,
        "team_balance": 90,
        "experience_rating": 95,
    },
    "Chennai Super Kings": {
        "squad": [
            "MS Dhoni", "Ruturaj Gaikwad", "Devon Conway", "Ajinkya Rahane",
            "Shivam Dube", "Ambati Rayudu", "Ravindra Jadeja", "Moeen Ali",
            "Deepak Chahar", "Tushar Deshpande", "Matheesha Pathirana",
            "Maheesh Theekshana", "Simarjeet Singh", "Prashant Solanki", "Ben Stokes",
        ],
        "playing_xi": [
            "Ruturaj Gaikwad", "Devon Conway", "Ajinkya Rahane", "Shivam Dube",
            "MS Dhoni", "Ravindra Jadeja", "Moeen Ali", "Deepak Chahar",
            "Tushar Deshpande", "Matheesha Pathirana", "Maheesh Theekshana",
        ],
        "captain": "Ruturaj Gaikwad",
        "key_players": ["MS Dhoni", "Ruturaj Gaikwad", "Ravindra Jadeja"],
        "strengths": "Brilliant team culture, smart captaincy, reliable openers",
        "weaknesses": "Aging squad, pace bowling depth",
        "batting_rating": 85,
        "bowling_rating": 82,
        "team_balance": 88,
        "experience_rating": 93,
    },
    "Royal Challengers Bengaluru": {
        "squad": [
            "Virat Kohli", "Faf du Plessis", "Glenn Maxwell", "Rajat Patidar",
            "Dinesh Karthik", "Anuj Rawat", "Mahipal Lomror", "Cameron Green",
            "Mohammed Siraj", "Josh Hazlewood", "Karn Sharma", "Wanindu Hasaranga",
            "Harshal Patel", "Shahbaz Ahmed", "Alzarri Joseph",
        ],
        "playing_xi": [
            "Virat Kohli", "Faf du Plessis", "Rajat Patidar", "Glenn Maxwell",
            "Cameron Green", "Dinesh Karthik", "Shahbaz Ahmed", "Wanindu Hasaranga",
            "Harshal Patel", "Mohammed Siraj", "Josh Hazlewood",
        ],
        "captain": "Faf du Plessis",
        "key_players": ["Virat Kohli", "Glenn Maxwell", "Mohammed Siraj"],
        "strengths": "Explosive top order, quality overseas players",
        "weaknesses": "Lower middle order fragile, death bowling consistency",
        "batting_rating": 91,
        "bowling_rating": 80,
        "team_balance": 83,
        "experience_rating": 90,
    },
    "Kolkata Knight Riders": {
        "squad": [
            "Shreyas Iyer", "Phil Salt", "Sunil Narine", "Venkatesh Iyer",
            "Rinku Singh", "Manish Pandey", "Andre Russell", "Nitish Rana",
            "Varun Chakravarthy", "Harshit Rana", "Mitchell Starc",
            "Lockie Ferguson", "Anukul Roy", "Suyash Sharma", "Rahmanullah Gurbaz",
        ],
        "playing_xi": [
            "Phil Salt", "Sunil Narine", "Shreyas Iyer", "Venkatesh Iyer",
            "Rinku Singh", "Andre Russell", "Nitish Rana", "Varun Chakravarthy",
            "Harshit Rana", "Mitchell Starc", "Lockie Ferguson",
        ],
        "captain": "Shreyas Iyer",
        "key_players": ["Sunil Narine", "Andre Russell", "Mitchell Starc"],
        "strengths": "All-round balance, power hitters, mystery spinners",
        "weaknesses": "Inconsistent middle order",
        "batting_rating": 86,
        "bowling_rating": 87,
        "team_balance": 89,
        "experience_rating": 88,
    },
    "Rajasthan Royals": {
        "squad": [
            "Sanju Samson", "Jos Buttler", "Yashasvi Jaiswal", "Devdutt Padikkal",
            "Shimron Hetmyer", "Dhruv Jurel", "Ravichandran Ashwin", "Trent Boult",
            "Yuzvendra Chahal", "Prasidh Krishna", "Jason Holder",
            "Kuldeep Sen", "Sandeep Sharma", "Riyan Parag", "Tom Kohler-Cadmore",
        ],
        "playing_xi": [
            "Yashasvi Jaiswal", "Jos Buttler", "Sanju Samson", "Devdutt Padikkal",
            "Shimron Hetmyer", "Riyan Parag", "Ravichandran Ashwin", "Jason Holder",
            "Yuzvendra Chahal", "Trent Boult", "Prasidh Krishna",
        ],
        "captain": "Sanju Samson",
        "key_players": ["Yashasvi Jaiswal", "Jos Buttler", "Yuzvendra Chahal"],
        "strengths": "Explosive openers, quality spin attack, IPL-savvy core",
        "weaknesses": "Death bowling, no-name middle order",
        "batting_rating": 89,
        "bowling_rating": 85,
        "team_balance": 87,
        "experience_rating": 86,
    },
    "Sunrisers Hyderabad": {
        "squad": [
            "Pat Cummins", "Heinrich Klaasen", "Travis Head", "Aiden Markram",
            "Abdul Samad", "Glenn Phillips", "Mayank Agarwal", "Bhuvneshwar Kumar",
            "T Natarajan", "Umran Malik", "Jaydev Unadkat",
            "Washington Sundar", "Marco Jansen", "Rahul Tripathi", "Fazalhaq Farooqi",
        ],
        "playing_xi": [
            "Travis Head", "Mayank Agarwal", "Aiden Markram", "Heinrich Klaasen",
            "Abdul Samad", "Glenn Phillips", "Washington Sundar", "Pat Cummins",
            "Bhuvneshwar Kumar", "T Natarajan", "Fazalhaq Farooqi",
        ],
        "captain": "Pat Cummins",
        "key_players": ["Travis Head", "Heinrich Klaasen", "Pat Cummins"],
        "strengths": "Devastating batting lineup, pace variety",
        "weaknesses": "Spin bowling, lower batting order fragility",
        "batting_rating": 93,
        "bowling_rating": 81,
        "team_balance": 85,
        "experience_rating": 84,
    },
    "Delhi Capitals": {
        "squad": [
            "David Warner", "Prithvi Shaw", "Mitchell Marsh", "Rishabh Pant",
            "Axar Patel", "Rovman Powell", "Kuldeep Yadav", "Anrich Nortje",
            "Mustafizur Rahman", "Ishant Sharma", "Lalit Yadav",
            "Yash Dhull", "Rilee Rossouw", "Pravin Dubey", "Vicky Ostwal",
        ],
        "playing_xi": [
            "David Warner", "Prithvi Shaw", "Mitchell Marsh", "Rishabh Pant",
            "Rovman Powell", "Axar Patel", "Lalit Yadav", "Kuldeep Yadav",
            "Anrich Nortje", "Mustafizur Rahman", "Ishant Sharma",
        ],
        "captain": "Rishabh Pant",
        "key_players": ["Rishabh Pant", "Axar Patel", "Kuldeep Yadav"],
        "strengths": "Dynamic wicket-keeper batter, quality spinners",
        "weaknesses": "Inconsistent top order, over-reliance on Pant",
        "batting_rating": 84,
        "bowling_rating": 83,
        "team_balance": 82,
        "experience_rating": 85,
    },
    "Punjab Kings": {
        "squad": [
            "Shikhar Dhawan", "Jonny Bairstow", "Liam Livingstone", "Sam Curran",
            "Prabhsimran Singh", "Shahrukh Khan", "Harpreet Brar", "Arshdeep Singh",
            "Kagiso Rabada", "Nathan Ellis", "Rahul Chahar",
            "Rishi Dhawan", "Jitesh Sharma", "Bhanuka Rajapaksa", "Chris Woakes",
        ],
        "playing_xi": [
            "Shikhar Dhawan", "Jonny Bairstow", "Liam Livingstone", "Sam Curran",
            "Shahrukh Khan", "Prabhsimran Singh", "Harpreet Brar", "Arshdeep Singh",
            "Kagiso Rabada", "Nathan Ellis", "Rahul Chahar",
        ],
        "captain": "Shikhar Dhawan",
        "key_players": ["Liam Livingstone", "Kagiso Rabada", "Arshdeep Singh"],
        "strengths": "Explosive batting, quality overseas pacers",
        "weaknesses": "Vulnerable spin department, no settled middle order",
        "batting_rating": 82,
        "bowling_rating": 82,
        "team_balance": 79,
        "experience_rating": 82,
    },
    "Gujarat Titans": {
        "squad": [
            "Shubman Gill", "Wriddhiman Saha", "Vijay Shankar", "Hardik Pandya",
            "David Miller", "Rahul Tewatia", "Mohammed Shami", "Rashid Khan",
            "Lockie Ferguson", "Alzarri Joseph", "Noor Ahmad",
            "Sai Sudharsan", "Darshan Nalkande", "Yash Dayal", "B Sai Sudharsan",
        ],
        "playing_xi": [
            "Shubman Gill", "Wriddhiman Saha", "Sai Sudharsan", "David Miller",
            "Vijay Shankar", "Rahul Tewatia", "Rashid Khan", "Noor Ahmad",
            "Mohammed Shami", "Alzarri Joseph", "Yash Dayal",
        ],
        "captain": "Shubman Gill",
        "key_players": ["Shubman Gill", "Rashid Khan", "Mohammed Shami"],
        "strengths": "Consistent batting, world-class spinner, lethal pace",
        "weaknesses": "Depend on Gill heavily, spin depth",
        "batting_rating": 87,
        "bowling_rating": 88,
        "team_balance": 88,
        "experience_rating": 83,
    },
    "Lucknow Super Giants": {
        "squad": [
            "KL Rahul", "Quinton de Kock", "Deepak Hooda", "Marcus Stoinis",
            "Nicholas Pooran", "Ayush Badoni", "Krishnappa Gowtham", "Jason Holder",
            "Avesh Khan", "Ravi Bishnoi", "Yudhvir Singh",
            "Mohsin Khan", "Mark Wood", "Kyle Mayers", "Amit Mishra",
        ],
        "playing_xi": [
            "KL Rahul", "Quinton de Kock", "Deepak Hooda", "Marcus Stoinis",
            "Nicholas Pooran", "Ayush Badoni", "Krishnappa Gowtham", "Ravi Bishnoi",
            "Avesh Khan", "Mark Wood", "Mohsin Khan",
        ],
        "captain": "KL Rahul",
        "key_players": ["KL Rahul", "Nicholas Pooran", "Ravi Bishnoi"],
        "strengths": "Top-order solidity, explosive overseas finishers",
        "weaknesses": "Death bowling, inconsistent spinners",
        "batting_rating": 85,
        "bowling_rating": 80,
        "team_balance": 81,
        "experience_rating": 84,
    },
}


def get_squads_df() -> pd.DataFrame:
    rows = []
    for team, data in SQUADS_2026.items():
        rows.append({
            "team": team,
            "captain": data["captain"],
            "key_players": ", ".join(data["key_players"]),
            "strengths": data["strengths"],
            "weaknesses": data["weaknesses"],
            "batting_rating": data["batting_rating"],
            "bowling_rating": data["bowling_rating"],
            "team_balance": data["team_balance"],
            "experience_rating": data["experience_rating"],
            "squad_size": len(data["squad"]),
            "titles": TITLES.get(team, 0),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Generating IPL match data...")
    matches = generate_match_data()
    print(f"  Total matches generated: {len(matches)}")

    print("Computing team season stats...")
    stats = compute_team_season_stats(matches)
    print(f"  Total team-season rows: {len(stats)}")

    print("Computing head-to-head...")
    h2h = compute_h2h(matches)
    print(f"  H2H combinations: {len(h2h)}")

    print("Building squad dataset...")
    squads = get_squads_df()
    print(squads[["team", "batting_rating", "bowling_rating", "team_balance"]].to_string())
    print("\n✅ Data generation complete.")
