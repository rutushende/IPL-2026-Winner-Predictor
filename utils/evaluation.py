"""
=============================================================================
  IPL 2026 PREDICTION PROJECT
  Module: utils/evaluation.py
  Purpose: Evaluation metrics, confusion matrix, ROC curves, feature
           importance plots, and model performance visualizations.
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, classification_report
)
import os

PALETTE = ["#FF6B35", "#004E89", "#1A936F", "#C62828", "#6A0572",
           "#0277BD", "#00838F", "#558B2F", "#F57F17", "#4527A0"]

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ─────────────────────────────────────────────
# 1. CONFUSION MATRIX PLOT
# ─────────────────────────────────────────────
def plot_confusion_matrix(cm: np.ndarray, model_name: str, save_path: str = None):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Team2 Wins", "Team1 Wins"],
                yticklabels=["Team2 Wins", "Team1 Wins"],
                ax=ax, linewidths=0.5)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# 2. ROC CURVE PLOT
# ─────────────────────────────────────────────
def plot_roc_curves(models_results: list, X_test, y_test, save_path: str = None):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")

    for i, r in enumerate(models_results):
        model = r["model"]
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=PALETTE[i], linewidth=2,
                    label=f"{r['model_name']} (AUC={roc_auc:.3f})")

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — Model Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# 3. MODEL COMPARISON BAR CHART
# ─────────────────────────────────────────────
def plot_model_comparison(results: list, save_path: str = None):
    metrics = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
    model_names = [r["model_name"] for r in results]
    x = np.arange(len(metrics))
    width = 0.8 / len(results)

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, r in enumerate(results):
        vals = [r[m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=r["model_name"],
                      color=PALETTE[i], alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width * (len(results) - 1) / 2)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"], fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# 4. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "t1_strength", "t1_batting", "t1_bowling", "t1_experience",
    "t1_form", "t1_toss_adv", "t1_balance", "t1_titles",
    "t2_strength", "t2_batting", "t2_bowling", "t2_experience",
    "t2_form", "t2_toss_adv", "t2_balance", "t2_titles",
    "h2h_t1_win_rate", "strength_diff", "batting_diff",
    "bowling_diff", "form_diff", "home_adv", "toss_winner_is_t1",
]

def plot_feature_importance(best_result: dict, save_path: str = None):
    model = best_result["model"]

    # Extract underlying estimator if pipeline
    estimator = model
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_importances_") or hasattr(step, "coef_"):
                estimator = step
                break

    fig, ax = plt.subplots(figsize=(9, 6))

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
        idx = np.argsort(importances)
        ax.barh(np.array(FEATURE_COLS)[idx], importances[idx],
                color=PALETTE[1], alpha=0.85, edgecolor="white")
        ax.set_title(f"Feature Importance — {best_result['model_name']}",
                     fontsize=13, fontweight="bold")

    elif hasattr(estimator, "coef_"):
        importances = np.abs(estimator.coef_[0])
        idx = np.argsort(importances)
        ax.barh(np.array(FEATURE_COLS)[idx], importances[idx],
                color=PALETTE[0], alpha=0.85, edgecolor="white")
        ax.set_title(f"Feature Coefficients — {best_result['model_name']}",
                     fontsize=13, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "Feature importance not available\nfor this model type",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)

    ax.set_xlabel("Importance Score", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# 5. WIN PROBABILITY BAR CHART
# ─────────────────────────────────────────────
def plot_win_probabilities(team_profiles: pd.DataFrame, save_path: str = None):
    df = team_profiles.sort_values("win_probability", ascending=True)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(df))]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df["team"], df["win_probability"], color=colors,
                   edgecolor="white", alpha=0.9, height=0.65)

    for bar, val in zip(bars, df["win_probability"]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left", fontsize=10, fontweight="bold")

    ax.set_xlabel("Win Probability (%)", fontsize=11)
    ax.set_title("IPL 2026 — Predicted Win Probabilities", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(0, df["win_probability"].max() + 3)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# 6. TEAM RADAR CHART
# ─────────────────────────────────────────────
def plot_team_radar(team_profiles: pd.DataFrame, top_n: int = 4, save_path: str = None):
    categories = ["Batting", "Bowling", "Experience", "Form", "Balance"]
    top_teams = team_profiles.head(top_n)

    from data.data_generator import SQUADS_2026, TITLES
    from utils.feature_engineering import (
        compute_batting_index, compute_bowling_index,
        compute_experience_score
    )

    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, axes = plt.subplots(1, top_n, figsize=(4 * top_n, 4),
                              subplot_kw=dict(polar=True))
    if top_n == 1:
        axes = [axes]

    for i, (_, row) in enumerate(top_teams.iterrows()):
        team = row["team"]
        squad_row = SQUADS_2026.get(team, {})
        vals = [
            squad_row.get("batting_rating", 80) / 100,
            squad_row.get("bowling_rating", 80) / 100,
            squad_row.get("experience_rating", 80) / 100,
            row["form_score"],
            squad_row.get("team_balance", 80) / 100,
        ]
        vals += vals[:1]

        ax = axes[i]
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(0)
        ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=8)
        ax.plot(angles, vals, "o-", linewidth=2, color=PALETTE[i])
        ax.fill(angles, vals, alpha=0.25, color=PALETTE[i])
        ax.set_ylim(0, 1)
        ax.set_title(f"{team}\n({row['win_probability']:.1f}%)",
                     size=9, fontweight="bold", pad=15)

    fig.suptitle("Top Teams — Radar Analysis", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# 7. HISTORICAL WIN TREND
# ─────────────────────────────────────────────
def plot_historical_trends(stats_df: pd.DataFrame, teams: list = None, save_path: str = None):
    if teams is None:
        from data.data_generator import TEAMS
        teams = TEAMS[:6]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, team in enumerate(teams):
        td = stats_df[stats_df["team"] == team].sort_values("season")
        ax.plot(td["season"], td["win_pct"], marker="o", linewidth=2,
                color=PALETTE[i], label=team, alpha=0.85, markersize=4)

    ax.set_xlabel("Season", fontsize=11)
    ax.set_ylabel("Win Percentage", fontsize=11)
    ax.set_title("Win Percentage Trends (2008–2025)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    print("Evaluation module loaded successfully.")
    print("Available plots: confusion_matrix, roc_curves, model_comparison,")
    print("                 feature_importance, win_probabilities, radar, trends")
