"""
=============================================================================
  IPL 2026 PREDICTION PROJECT
  Module: models/train_models.py
  Purpose: Train and compare Logistic Regression, Random Forest,
           XGBoost, and Neural Network (MLP) classifiers.
           Returns the best model + all evaluation metrics.
=============================================================================

  MODEL COMPARISON
  ────────────────
  Model              | Pros                          | Cons
  ─────────────────────────────────────────────────────────
  Logistic Reg.      | Interpretable, fast           | Assumes linearity
  Random Forest      | Handles non-linearity, robust | Slow for large data
  XGBoost            | Best accuracy, gradient boost | Needs tuning
  Neural Net (MLP)   | Captures complex patterns     | Needs more data

  RECOMMENDATION: XGBoost for best accuracy,
                  Logistic Regression for interpretability.
=============================================================================
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.pipeline import Pipeline
import joblib

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️  XGBoost not installed. Run: pip install xgboost")


# ─────────────────────────────────────────────
# FEATURE COLUMNS
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "t1_strength", "t1_batting", "t1_bowling", "t1_experience",
    "t1_form", "t1_toss_adv", "t1_balance", "t1_titles",
    "t2_strength", "t2_batting", "t2_bowling", "t2_experience",
    "t2_form", "t2_toss_adv", "t2_balance", "t2_titles",
    "h2h_t1_win_rate", "strength_diff", "batting_diff",
    "bowling_diff", "form_diff", "home_adv", "toss_winner_is_t1",
]
TARGET_COL = "target"


# ─────────────────────────────────────────────
# DATA PREP
# ─────────────────────────────────────────────
def prepare_data(features_df: pd.DataFrame):
    """Split into train/test with stratification."""
    df = features_df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    print(f"  Class balance (train) — Win: {y_train.sum()} | Loss: {(1-y_train).sum()}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# DEFINE MODELS
# ─────────────────────────────────────────────
def get_models():
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0, max_iter=1000, random_state=42, solver="lbfgs"
            )),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        ),
        "Neural Network (MLP)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(64, 32, 16),
                activation="relu",
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
            )),
        ]),
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )
    return models


# ─────────────────────────────────────────────
# EVALUATE ONE MODEL
# ─────────────────────────────────────────────
def evaluate_model(model, X_train, X_test, y_train, y_test, name: str) -> dict:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
    cm = confusion_matrix(y_test, y_pred)

    result = {
        "model_name": name,
        "model": model,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "auc_roc": round(auc, 4),
        "cv_mean": round(cv_scores.mean(), 4),
        "cv_std": round(cv_scores.std(), 4),
        "confusion_matrix": cm,
    }

    print(f"\n  ─── {name} ───")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}  |  Recall: {rec:.4f}  |  F1: {f1:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"  CV (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Confusion Matrix:\n{cm}")

    return result


# ─────────────────────────────────────────────
# TRAIN ALL MODELS
# ─────────────────────────────────────────────
def train_all_models(features_df: pd.DataFrame) -> tuple:
    """
    Train all models. Returns (results_list, best_model, X_test, y_test).
    """
    print("\n" + "="*60)
    print("  MODEL TRAINING & EVALUATION")
    print("="*60)

    X_train, X_test, y_train, y_test = prepare_data(features_df)
    models = get_models()
    results = []

    for name, model in models.items():
        r = evaluate_model(model, X_train, X_test, y_train, y_test, name)
        results.append(r)

    # Best model by F1-score
    best = max(results, key=lambda x: x["f1_score"])
    print(f"\n✅ Best Model: {best['model_name']} (F1={best['f1_score']:.4f})")

    return results, best, X_test, y_test


# ─────────────────────────────────────────────
# COMPARISON TABLE
# ─────────────────────────────────────────────
def model_comparison_table(results: list) -> pd.DataFrame:
    rows = [{
        "Model": r["model_name"],
        "Accuracy": r["accuracy"],
        "Precision": r["precision"],
        "Recall": r["recall"],
        "F1-Score": r["f1_score"],
        "AUC-ROC": r["auc_roc"],
        "CV Mean±Std": f"{r['cv_mean']:.4f}±{r['cv_std']:.4f}",
    } for r in results]
    df = pd.DataFrame(rows).sort_values("F1-Score", ascending=False)
    return df


# ─────────────────────────────────────────────
# SAVE MODEL
# ─────────────────────────────────────────────
def save_best_model(best_result: dict, path: str = "outputs/best_model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(best_result["model"], path)
    print(f"💾 Model saved → {path}")


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data.data_generator import (
        generate_match_data, compute_team_season_stats,
        compute_h2h, get_squads_df
    )
    from utils.feature_engineering import build_match_features

    print("Generating data...")
    matches = generate_match_data()
    stats = compute_team_season_stats(matches)
    h2h = compute_h2h(matches)
    squads = get_squads_df()
    features = build_match_features(matches, stats, h2h, squads)

    results, best, X_test, y_test = train_all_models(features)

    print("\n📊 Model Comparison Table:")
    print(model_comparison_table(results).to_string(index=False))

    save_best_model(best, "../outputs/best_model.pkl")
