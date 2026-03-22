"""
Configuration management.
Loads config/config.yaml and exposes typed dataclasses.
All magic numbers live here — nowhere else in the codebase.
"""

from __future__ import annotations
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"


@dataclass(frozen=True)
class ArchetypeConfig:
    share: float
    daily_txn_rate: float
    avg_value: float
    churn_prob: float
    p2m_ratio: float


@dataclass(frozen=True)
class SimulationConfig:
    n_users: int
    n_days: int
    seed: int
    archetypes: Dict[str, ArchetypeConfig]
    categories: List[str]
    city_tier_dist: List[float]
    age_groups: List[str]
    age_dist: List[float]


@dataclass(frozen=True)
class FeatureConfig:
    early_window_days: int
    numeric: List[str]
    binary: List[str]
    target: str

    @property
    def all_features(self) -> List[str]:
        return self.numeric + self.binary


@dataclass(frozen=True)
class LGBMConfig:
    n_estimators: int
    learning_rate: float
    num_leaves: int
    min_child_samples: int
    subsample: float
    colsample_bytree: float
    early_stopping_rounds: int


@dataclass(frozen=True)
class LogisticConfig:
    C: float
    max_iter: int


@dataclass(frozen=True)
class ChurnModelConfig:
    test_size: float
    cv_folds: int
    lgbm: LGBMConfig
    logistic: LogisticConfig


@dataclass(frozen=True)
class UpliftModelConfig:
    treatment_fraction: float
    classification_threshold: float
    n_estimators: int


@dataclass(frozen=True)
class ModelConfig:
    churn: ChurnModelConfig
    uplift: UpliftModelConfig


@dataclass(frozen=True)
class InterventionConfig:
    default_cashback: int
    default_budget: int
    response_rates: Dict[str, float]


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    format: str
    datefmt: str


@dataclass(frozen=True)
class Config:
    simulation: SimulationConfig
    features: FeatureConfig
    model: ModelConfig
    intervention: InterventionConfig
    logging: LoggingConfig


def load_config(path: Path = CONFIG_PATH) -> Config:
    """Load and validate configuration from YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    sim_raw = raw["simulation"]
    archetypes = {
        name: ArchetypeConfig(**params)
        for name, params in sim_raw["archetypes"].items()
    }

    simulation = SimulationConfig(
        n_users=sim_raw["n_users"],
        n_days=sim_raw["n_days"],
        seed=sim_raw["seed"],
        archetypes=archetypes,
        categories=sim_raw["categories"],
        city_tier_dist=sim_raw["city_tier_dist"],
        age_groups=sim_raw["age_groups"],
        age_dist=sim_raw["age_dist"],
    )

    feat_raw = raw["features"]
    features = FeatureConfig(
        early_window_days=feat_raw["early_window_days"],
        numeric=feat_raw["numeric"],
        binary=feat_raw["binary"],
        target=feat_raw["target"],
    )

    model_raw = raw["model"]
    churn_raw = model_raw["churn"]
    model = ModelConfig(
        churn=ChurnModelConfig(
            test_size=churn_raw["test_size"],
            cv_folds=churn_raw["cv_folds"],
            lgbm=LGBMConfig(**churn_raw["lgbm"]),
            logistic=LogisticConfig(**churn_raw["logistic"]),
        ),
        uplift=UpliftModelConfig(**model_raw["uplift"]),
    )

    iv_raw = raw["intervention"]
    intervention = InterventionConfig(
        default_cashback=iv_raw["default_cashback"],
        default_budget=iv_raw["default_budget"],
        response_rates=iv_raw["response_rates"],
    )

    log_raw = raw["logging"]
    logging_cfg = LoggingConfig(
        level=log_raw["level"],
        format=log_raw["format"],
        datefmt=log_raw["datefmt"],
    )

    return Config(
        simulation=simulation,
        features=features,
    