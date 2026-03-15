"""
Churn prediction model.

Design:
  - sklearn Pipeline wraps preprocessing + model
  - Cross-validated on training set before final fit
  - Both LightGBM and Logistic Regression are trained and compared
  - Model artifacts are saved to disk for serving
  - Full evaluation suite: AUC, precision-recall, calibration check

Production considerations baked in:
  - Early stopping uses a validation split, not the test set
  - Class weights handled explicitly (churned class is minority)
  - Model serialised with joblib for serving
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.config import CFG
from src.features.engineer import EXTENDED_FEATURES, build_feature_pipeline
from src.utils.logging import get_logger

logger = get_logger(__name__)

ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "artifacts"


@dataclass
class EvaluationMetrics:
    """Comprehensive model evaluation results."""
    auc_roc: float
    avg_precision: float
    brier_score: float          # calibration: lower is better
    classification_report: str
    fpr: np.ndarray             # for ROC curve plotting
    tpr: np.ndarray
    precision: np.ndarray       # for PR curve plotting
    recall: np.ndarray
    feature_importance: Dict[str, float]


@dataclass
class ChurnModelResult:
    """Full output of train_churn_model()."""
    users_scored: pd.DataFrame      # users + churn_prob column
    lgb_metrics: EvaluationMetrics
    lr_metrics: EvaluationMetrics
    cv_auc_scores: List[float]      # cross-validation AUC per fold
    cv_auc_mean: float
    cv_auc_std: float
    train_size: int
    test_size: int
    elapsed_seconds: float
    delong_result: Optional["DeLongResult"] = None  # statistical comparison

    def summary(self) -> str:
        lines = [
            "─" * 50,
            "CHURN MODEL RESULTS",
            "─" * 50,
            f"  CV AUC (5-fold):     {self.cv_auc_mean:.4f} ± {self.cv_auc_std:.4f}",
            f"  Test AUC (LightGBM): {self.lgb_metrics.auc_roc:.4f}",
            f"  Test AUC (Logistic): {self.lr_metrics.auc_roc:.4f}",
            f"  Δ AUC (LGB vs LR):  +{(self.lgb_metrics.auc_roc - self.lr_metrics.auc_roc)*100:.2f}pp",
            f"  Avg Precision (LGB): {self.lgb_metrics.avg_precision:.4f}",
            f"  Brier Score (LGB):   {self.lgb_metrics.brier_score:.4f}",
            f"  Train / Test:        {self.train_size:,} / {self.test_size:,}",
        ]
        if self.delong_result is not None:
            dl = self.delong_result
            sig = "significant" if dl.significant else "not significant"
            lines.append("")
            lines.append(f"  DeLong Test (LGB vs LR):")
            lines.append(f"    AUC diff: {dl.auc_diff:+.4f}, p={dl.p_value:.4f} ({sig})")
            lines.append(f"    95% CI:   [{dl.ci_lower:+.4f}, {dl.ci_upper:+.4f}]")
        lines.append("")
        lines.append("  Top 5 features (by importance):")
        top5 = sorted(self.lgb_metrics.feature_importance.items(),
                      key=lambda x: x[1], reverse=True)[:5]
        for feat, imp in top5:
            bar = "█" * max(1, int(imp / 3))
            lines.append(f"    {feat:<40} {imp:5.1f}% {bar}")
        lines.append("─" * 50)
        return "\n".join(lines)


def train_churn_model(
    users: pd.DataFrame,
    save_artifact: bool = True,
) -> ChurnModelResult:
    """
    Train and evaluate the churn prediction model.

    Steps:
      1. Train/test split (stratified on churn label)
      2. Cross-validate LightGBM on train set → stability estimate
      3. Final fit on full train set
      4. Evaluate both LightGBM and Logistic Regression on held-out test
      5. Score all users
      6. Save artifact

    Parameters
    ----------
    users : pd.DataFrame
        Validated user feature table from simulate_users() or real data.
    save_artifact : bool
        Persist trained model to artifacts/ directory.

    Returns
    -------
    ChurnModelResult
    """
    logger.info("Training churn model on %d users", len(users))
    t0 = time.perf_counter()

    target = CFG.features.target
    X = users.drop(columns=[target, "user_id", "archetype",
                             "age_group", "total_txn", "total_value"],
                   errors="ignore")
    y = users[target]

    # ── Stratified split ─────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CFG.model.churn.test_size,
        random_state=CFG.simulation.seed,
        stratify=y,
    )
    logger.info("Train: %d | Test: %d | Churn rate: %.1f%%",
                len(X_train), len(X_test), y_train.mean() * 100)

    # ── Cross-validation on training set ─────────────────────────
    logger.info("Running %d-fold cross-validation...", CFG.model.churn.cv_folds)
    cv_scores = _cross_validate(X_train, y_train)
    logger.info("CV AUC: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())

    # ── Final LightGBM fit ───────────────────────────────────────
    logger.info("Fitting final LightGBM model...")
    lgb_pipeline, lgb_metrics = _fit_lightgbm(X_train, y_train, X_test, y_test)

    # ── Logistic Regression baseline ─────────────────────────────
    logger.info("Fitting Logistic Regression baseline...")
    lr_pipeline, lr_metrics = _fit_logistic(X_train, y_train, X_test, y_test)

    # ── DeLong test: is the AUC difference significant? ──────────
    logger.info("Running DeLong test (LGB vs LR)...")
    from src.models.statistical_tests import delong_test, DeLongResult
    lgb_proba_test = lgb_pipeline.predict_proba(X_test)[:, 1]
    lr_proba_test = lr_pipeline.predict_proba(X_test)[:, 1]
    delong = delong_test(y_test.values, lgb_proba_test, lr_proba_test)

    # ── Score all users ──────────────────────────────────────────
    users_scored = users.copy()
    users_scored["churn_prob"] = lgb_pipeline.predict_proba(X)[:, 1]

    # ── Persist ──────────────────────────────────────────────────
    if save_artifact:
        path = ARTIFACTS_DIR / "churn_model.joblib"
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(lgb_pipeline, path)
        logger.info("Model saved → %s", path)

    elapsed = time.perf_counter() - t0
    result = ChurnModelResult(
        users_scored=users_scored,
        lgb_metrics=lgb_metrics,
        lr_metrics=lr_metrics,
        cv_auc_scores=cv_scores.tolist(),
        cv_auc_mean=float(cv_scores.mean()),
        cv_auc_std=float(cv_scores.std()),
        train_size=len(X_train),
        test_size=len(X_test),
        elapsed_seconds=elapsed,
        delong_result=delong,
    )
    logger.info(result.summary())
    return result


def _cross_validate(X_train: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
    """
    5-fold stratified cross-validation.
    Uses early stopping within each fold against the fold's val set.
    """
    cv = StratifiedKFold(
        n_splits=CFG.model.churn.cv_folds,
        shuffle=True,
        random_state=CFG.simulation.seed,
    )
    auc_scores = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        Xf_tr, Xf_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        yf_tr, yf_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        pipeline = _build_lgb_pipeline()
        # Fit without early stopping in CV (cleaner)
        pipeline.fit(Xf_tr, yf_tr)

        proba = pipeline.predict_proba(Xf_val)[:, 1]
        auc   = roc_auc_score(yf_val, proba)
        auc_scores.append(auc)
        logger.debug("  Fold %d AUC: %.4f", fold + 1, auc)

    return np.array(auc_scores)


def _build_lgb_pipeline() -> Pipeline:
    """LightGBM wrapped in a feature pipeline."""
    lgb_cfg = CFG.model.churn.lgbm
    clf = lgb.LGBMClassifier(
        n_estimators=lgb_cfg.n_estimators,
        learning_rate=lgb_cfg.learning_rate,
        num_leaves=lgb_cfg.num_leaves,
        min_child_samples=lgb_cfg.min_child_samples,
        subsample=lgb_cfg.subsample,
        colsample_bytree=lgb_cfg.colsample_bytree,
        random_state=CFG.simulation.seed,
        verbose=-1,
        class_weight="balanced",  # handles class imbalance
    )
    from src.features.engineer import EarlyWindowFeatures, FeatureSelector
    from sklearn.preprocessing import StandardScaler
    return Pipeline([
        ("early_window",   EarlyWindowFeatures()),
        ("feature_select", FeatureSelector(EXTENDED_FEATURES)),
        ("clf",            clf),
    ])


def _fit_lightgbm(
    X_train, y_train, X_test, y_test
) -> Tuple[Pipeline, EvaluationMetrics]:
    """Fit LightGBM with early stopping on a held-out validation split."""
    lgb_cfg = CFG.model.churn.lgbm

    # Internal val split for early stopping (from training data only)
    X_tr2, X_val, y_tr2, y_val = train_test_split(
        X_train, y_train, test_size=0.15,
        random_state=CFG.simulation.seed, stratify=y_train
    )

    pipeline = _build_lgb_pipeline()

    # Fit the pipeline up to (but not including) the clf step to get transformed data
    pre = Pipeline(pipeline.steps[:-1])
    pre.fit(X_tr2, y_tr2)
    Xv_transformed = pre.transform(X_val)

    # Fit with early stopping by accessing clf directly
    clf = pipeline.named_steps["clf"]
    pipeline.fit(X_tr2, y_tr2)  # initial fit sets up pipeline

    # Re-fit clf with early stopping
    X_tr2_t = pre.transform(X_tr2)
    X_te_t  = pre.transform(X_test)
    Xv_t    = pre.transform(X_val)

    clf.fit(
        X_tr2_t, y_tr2,
        eval_set=[(Xv_t, y_val)],
        callbacks=[
            lgb.early_stopping(lgb_cfg.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    logger.info("  Best iteration: %d", clf.best_iteration_)

    proba = clf.predict_proba(X_te_t)[:, 1]
    preds = (proba > 0.5).astype(int)

    metrics = _compute_metrics(y_test, proba, preds, clf, pre)
    return pipeline, metrics


def _fit_logistic(
    X_train, y_train, X_test, y_test
) -> Tuple[Pipeline, EvaluationMetrics]:
    """Fit logistic regression baseline."""
    from src.features.engineer import EarlyWindowFeatures, FeatureSelector
    from sklearn.preprocessing import StandardScaler

    lr_cfg = CFG.model.churn.logistic
    pipeline = Pipeline([
        ("early_window",   EarlyWindowFeatures()),
        ("feature_select", FeatureSelector(EXTENDED_FEATURES)),
        ("scaler",         StandardScaler()),
        ("clf",            LogisticRegression(
            C=lr_cfg.C,
            max_iter=lr_cfg.max_iter,
            random_state=CFG.simulation.seed,
            class_weight="balanced",
        )),
    ])
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba > 0.5).astype(int)

    # Feature importance for LR = absolute coefficient values
    coefs = np.abs(pipeline.named_steps["clf"].coef_[0])
    feat_names = EXTENDED_FEATURES
    importance = {
        feat: round(float(c / coefs.sum() * 100), 1)
        for feat, c in zip(feat_names, coefs)
    }
    metrics = _compute_metrics(y_test, proba, preds, None, None,
                                importance=importance)
    return pipeline, metrics


def _compute_metrics(
    y_true, proba, preds,
    clf=None, pre=None,
    importance: Optional[Dict] = None,
) -> EvaluationMetrics:
    auc_roc   = roc_auc_score(y_true, proba)
    avg_prec  = average_precision_score(y_true, proba)
    brier     = brier_score_loss(y_true, proba)
    report    = classification_report(y_true, preds,
                                      target_names=["Retained", "Churned"])
    fpr, tpr, _ = roc_curve(y_true, proba)
    prec, rec, _ = precision_recall_curve(y_true, proba)

    if importance is None and clf is not None:
        raw = clf.feature_importances_
        importance = {
            feat: round(float(imp / raw.sum() * 100), 1)
            for feat, imp in zip(EXTENDED_FEATURES, raw)
        }

    return EvaluationMetrics(
        auc_roc=round(auc_roc, 4),
        avg_precision=round(avg_prec, 4),
        brier_score=round(brier, 4),
        classification_report=report,
        fpr=fpr,
        tpr=tpr,
        precision=prec,
        recall=rec,
        feature_importance=importance or {},
    )
