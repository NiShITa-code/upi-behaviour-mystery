"""
CSV data loader — Bring Your Own Data mode.

Validates and normalises user-uploaded CSV files to match the
schema expected by the pipeline. Provides clear, actionable
error messages when the data doesn't meet requirements.

Supports two input formats:
  1. Full format: all features pre-computed (plug and play)
  2. Minimal format: just user_id + transaction log → we compute features

Usage:
    from src.data.csv_loader import load_user_csv, load_transaction_csv
    users = load_user_csv("my_users.csv")
    # or
    users, txns = load_transaction_csv("my_transactions.csv")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.schema import validate_user_dataframe
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CSVValidationResult:
    """Result of CSV validation — either success with data, or failure with messages."""
    valid: bool
    users: Optional[pd.DataFrame] = None
    transactions: Optional[pd.DataFrame] = None
    warnings: List[str] = None
    errors: List[str] = None
    n_users: int = 0
    n_transactions: int = 0

    def __post_init__(self):
        self.warnings = self.warnings or []
        self.errors = self.errors or []


# ── Required columns for full-format CSV ────────────────────
REQUIRED_USER_COLS = {
    "user_id", "txn_d7", "value_d7", "txn_d14", "value_d14",
    "has_first_p2m_d14", "first_p2m_day", "cat_diversity",
    "p2m_ratio", "city_tier", "churned",
}

# ── Required columns for transaction-format CSV ─────────────
REQUIRED_TXN_COLS = {
    "user_id", "day", "category", "value",
}

# ── Optional columns (we'll fill defaults if missing) ────────
OPTIONAL_COLS = {
    "archetype": "Unknown",
    "age_group": "Unknown",
    "total_txn": 0,
    "total_value": 0.0,
}


def load_user_csv(file_or_path) -> CSVValidationResult:
    """
    Load and validate a user-level CSV (full format).

    Expected columns (minimum):
        user_id, txn_d7, value_d7, txn_d14, value_d14,
        has_first_p2m_d14, first_p2m_day, cat_diversity,
        p2m_ratio, city_tier, churned

    Optional:
        archetype, age_group, total_txn, total_value

    Parameters
    ----------
    file_or_path : str or file-like
        CSV file path or uploaded file object (from Streamlit).

    Returns
    -------
    CSVValidationResult
    """
    errors = []
    warnings = []

    try:
        df = pd.read_csv(file_or_path)
    except Exception as e:
        return CSVValidationResult(valid=False, errors=[f"Failed to read CSV: {e}"])

    if len(df) == 0:
        return CSVValidationResult(valid=False, errors=["CSV is empty (0 rows)."])

    # Check required columns
    missing = REQUIRED_USER_COLS - set(df.columns)
    if missing:
        return CSVValidationResult(
            valid=False,
            errors=[f"Missing required columns: {', '.join(sorted(missing))}"],
        )

    # Fill optional columns with defaults
    for col, default in OPTIONAL_COLS.items():
        if col not in df.columns:
            df[col] = default
            warnings.append(f"Column '{col}' not found — filled with default: {default}")

    # Type coercion
    try:
        df["user_id"] = df["user_id"].astype(int)
        df["churned"] = df["churned"].astype(int)
        df["city_tier"] = df["city_tier"].astype(int)
        df["has_first_p2m_d14"] = df["has_first_p2m_d14"].astype(int)
        for col in ["txn_d7", "txn_d14", "first_p2m_day", "cat_diversity"]:
            df[col] = df[col].astype(int)
        for col in ["value_d7", "value_d14", "p2m_ratio"]:
            df[col] = df[col].astype(float)
    except (ValueError, TypeError) as e:
        errors.append(f"Type conversion failed: {e}")

    # Validate with schema
    try:
        validate_user_dataframe(df)
    except ValueError as e:
        errors.append(f"Schema validation failed: {e}")

    if errors:
        return CSVValidationResult(valid=False, errors=errors, warnings=warnings)

    # Compute total_txn and total_value if not provided
    if (df["total_txn"] == 0).all():
        df["total_txn"] = df["txn_d14"]  # best approximation
        df["total_value"] = df["value_d14"]
        warnings.append("total_txn/total_value not provided — approximated from 14-day values")

    n_users = len(df)
    logger.info("CSV loaded: %d users, %d columns", n_users, len(df.columns))

    return CSVValidationResult(
        valid=True,
        users=df,
        warnings=warnings,
        n_users=n_users,
    )


def load_transaction_csv(file_or_path) -> CSVValidationResult:
    """
    Load a transaction-level CSV and compute user features.

    Expected columns:
        user_id, day, category, value
    Optional:
        is_p2m (0/1) — if not present, inferred from category

    This computes all required user-level features from raw
    transactions, matching the output of simulate_users().

    Parameters
    ----------
    file_or_path : str or file-like
        CSV file path or uploaded file object.

    Returns
    -------
    CSVValidationResult
    """
    errors = []
    warnings = []

    try:
        txns = pd.read_csv(file_or_path)
    except Exception as e:
        return CSVValidationResult(valid=False, errors=[f"Failed to read CSV: {e}"])

    if len(txns) == 0:
        return CSVValidationResult(valid=False, errors=["CSV is empty (0 rows)."])

    missing = REQUIRED_TXN_COLS - set(txns.columns)
    if missing:
        return CSVValidationResult(
            valid=False,
            errors=[f"Missing required columns: {', '.join(sorted(missing))}"],
        )

    # Infer is_p2m if not present
    if "is_p2m" not in txns.columns:
        p2p_keywords = ["p2p", "transfer", "send", "peer"]
        txns["is_p2m"] = txns["category"].apply(
            lambda c: 0 if any(kw in str(c).lower() for kw in p2p_keywords) else 1
        )
        warnings.append("'is_p2m' column not found — inferred from category names")

    # Compute user features from transactions
    users = _compute_features_from_transactions(txns)

    # Add churned column — default to 0, user can override
    if "churned" not in users.columns:
        users["churned"] = 0
        warnings.append(
            "'churned' label not available — set to 0 for all users. "
            "Upload a user CSV with 'churned' column for churn analysis."
        )

    # Fill optional columns
    if "archetype" not in txns.columns:
        users["archetype"] = "Unknown"
    if "city_tier" not in users.columns:
        users["city_tier"] = 1
        warnings.append("'city_tier' not available — defaulted to 1 (metro)")
    if "age_group" not in users.columns:
        users["age_group"] = "Unknown"

    n_users = len(users)
    n_txns = len(txns)
    logger.info("Transaction CSV loaded: %d users, %d transactions", n_users, n_txns)

    return CSVValidationResult(
        valid=True,
        users=users,
        transactions=txns,
        warnings=warnings,
        n_users=n_users,
        n_transactions=n_txns,
    )


def _compute_features_from_transactions(txns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute user-level features from raw transaction data.
    Mirrors the feature computation in simulate_users().
    """
    d7_mask = txns["day"] < 7
    d14_mask = txns["day"] < 14

    # Per-user aggregations
    user_features = []
    for uid, user_txns in txns.groupby("user_id"):
        d7 = user_txns[d7_mask.loc[user_txns.index]]
        d14 = user_txns[d14_mask.loc[user_txns.index]]

        # First merchant payment day
        p2m_txns = user_txns[user_txns["is_p2m"] == 1]
        first_p2m_day = int(p2m_txns["day"].min()) if len(p2m_txns) > 0 else 999

        user_features.append({
            "user_id": uid,
            "txn_d7": len(d7),
            "value_d7": round(d7["value"].sum(), 2),
            "txn_d14": len(d14),
            "value_d14": round(d14["value"].sum(), 2),
            "has_first_p2m_d14": int(first_p2m_day < 14),
            "first_p2m_day": first_p2m_day,
            "cat_diversity": user_txns["category"].nunique(),
            "p2m_ratio": round(
                user_txns["is_p2m"].mean() if len(user_txns) > 0 else 0.0, 4
            ),
            "total_txn": len(user_txns),
            "total_value": round(user_txns["value"].sum(), 2),
        })

    return pd.DataFrame(user_features)


def generate_sample_csv(n_users: int = 20, seed: int = 42) -> pd.DataFrame:
    """
    Generate a small sample CSV for users to download as a template.

    Returns a DataFrame matching the expected user CSV format
    with realistic-looking values.
    """
    rng = np.random.default_rng(seed)

    rows = []
    for uid in range(n_users):
        txn_d7 = int(rng.poisson(3))
        txn_d14 = txn_d7 + int(rng.poisson(4))
        value_d7 = round(txn_d7 * rng.lognormal(6.5, 0.5), 2) if txn_d7 > 0 else 0
        value_d14 = value_d7 + round(
            (txn_d14 - txn_d7) * rng.lognormal(6.5, 0.5), 2
        ) if txn_d14 > txn_d7 else value_d7

        has_p2m = int(rng.random() > 0.3)
        first_p2m = int(rng.integers(0, 14)) if has_p2m else 999

        rows.append({
            "user_id": uid + 1,
            "city_tier": int(rng.choice([1, 2, 3])),
            "txn_d7": txn_d7,
            "value_d7": value_d7,
            "txn_d14": txn_d14,
            "value_d14": value_d14,
            "has_first_p2m_d14": int(first_p2m < 14),
            "first_p2m_day": first_p2m,
            "cat_diversity": int(rng.integers(1, 8)),
            "p2m_ratio": round(float(rng.uniform(0.2, 0.8)), 2),
            "churned": int(rng.random() > 0.6),
        })

    return pd.DataFrame(rows)
