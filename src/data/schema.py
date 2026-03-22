"""
Data schemas — typed contracts for every data object in the pipeline.
Pydantic validates at runtime; mypy validates statically.

Why this matters in production:
  - Catches bad data immediately at ingestion, not 3 steps later
  - Self-documenting: the schema IS the data dictionary
  - Enables serialisation/deserialisation without boilerplate
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class UserRecord:
    """One row of the users table after feature engineering."""
    user_id: int
    archetype: str
    city_tier: int          # 1 = metro, 2 = tier-2, 3 = tier-3
    age_group: str

    # Early-window features (the predictive window)
    txn_d7: int             # transaction count, days 0–6
    value_d7: float         # total spend (₹), days 0–6
    txn_d14: int            # transaction count, days 0–13
    value_d14: float        # total spend (₹), days 0–13
    has_first_p2m_d14: int  # binary: first merchant payment within 14 days?
    first_p2m_day: int      # day of first merchant payment (999 = never)

    # Behaviour features
    cat_diversity: int      # distinct categories used (0–8)
    p2m_ratio: float        # fraction of txns that are merchant payments

    # Full-period summary (for analysis, not model training)
    total_txn: int
    total_value: float

    # Target
    churned: int            # 1 = went inactive within 12 months

    def validate(self) -> None:
        """Raise ValueError if record fails business logic constraints."""
        if not (0 <= self.p2m_ratio <= 1):
            raise ValueError(f"p2m_ratio {self.p2m_ratio} out of [0,1]")
        if self.city_tier not in (1, 2, 3):
            raise ValueError(f"city_tier {self.city_tier} not in {{1,2,3}}")
        if self.txn_d14 < self.txn_d7:
            raise ValueError("txn_d14 cannot be less than txn_d7")
        if self.churned not in (0, 1):
            raise ValueError(f"churned label {self.churned} not binary")


@dataclass
class TransactionRecord:
    """One row of the transactions table."""
    user_id: int
    archetype: str
    day: int                # day offset from registration (0 = day 1)
    category: str
    is_p2m: int             # 1 = merchant payment, 0 = P2P transfer
    value: float            # transaction value in ₹


@dataclass
class ChurnPrediction:
    """Output of the churn model for a single user."""
    user_id: int
    churn_prob: float       # P(churn) ∈ [0, 1]
    churn_label: int        # binarised at 0.5 threshold

    def __post_init__(self) -> None:
        if not (0.0 <= self.churn_prob <= 1.0):
            raise ValueError(f"churn_prob {self.churn_prob} out of [0,1]")


@dataclass
class UpliftPrediction:
    """Output of the uplift model for a single user."""
    user_id: int
    p0: float               # P(retain | no offer)
    p1: float               # P(retain | offer sent)
    uplift: float           # ITE = p1 - p0
    segment: str            # Persuadable | Sure Thing | Lost Cause | Sleeping Dog

    VALID_SEGMENTS = frozenset({
        "Persuadable", "Sure Thing", "Lost Cause", "Sleeping Dog"
    })

    def __post_init__(self) -> None:
        if self.segment not in self.VALID_SEGMENTS:
            raise ValueError(f"Unknown segment: {self.segment}")


def validate_user_dataframe(df: pd.DataFrame) -> None:
    """
    Validate a DataFrame of users against expected schema.
    Raises ValueError with clear message on first violation found.
    Called at the boundary between data layer and model layer.
    """
    required_cols = {
        "user_id", "archetype", "city_tier", "txn_d7", "value_d7",
        "txn_d14", "value_d14", "has_first_p2m_d14", "first_p2m_day",
        "cat_diversity", "p2m_ratio", "churned",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df["churned"].isin([0, 1]).sum() != len(df):
        raise ValueError("'churned' column must be binary (0/1)")

    if not df["p2m_ratio"].between(0, 1).all():
        bad = df[~df["p2m_ratio"].between(0, 1)]["p2m_ratio"].unique()
        raise ValueError(f"p2m_ratio out of [0,1]: {bad}")

    if not df["city_tier"].isin([1, 2, 3]).all():
        raise ValueError("city_tier must be 1, 2, or 3")

    if (df["txn_d14"] < df["txn_d7"]).any():
        raise ValueError("txn_d14 < txn_d7 for some rows — impossible")

    if df["user_id"].duplicated().any():
        raise ValueError("Duplicate user_ids found")

    if df.isnull().any().any