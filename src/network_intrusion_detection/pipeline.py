from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "demo_network_traffic.csv"


def load_or_create_demo_dataset(n_rows: int = 2000) -> pd.DataFrame:
    """Create or load a small demo dataset representing network traffic."""
    if DATASET_PATH.exists():
        return pd.read_csv(DATASET_PATH)

    rng = np.random.default_rng(42)
    proto_choices = ["tcp", "udp", "icmp", "arp"]
    service_choices = ["http", "dns", "smtp", "ftp", "ssh", "none"]
    state_choices = ["FIN", "INT", "CON", "REQ", "RST"]

    records = []
    for _ in range(n_rows):
        spkts = int(rng.integers(10, 500))
        dpkts = int(rng.integers(5, 600))
        sbytes = int(rng.integers(100, 60000))
        dbytes = int(rng.integers(80, 80000))
        dur = float(rng.uniform(0.01, 3.5))
        proto = rng.choice(proto_choices)
        service = rng.choice(service_choices)
        state = rng.choice(state_choices)

        suspicious = (
            (spkts > 350 and dpkts > 250)
            or (sbytes > 30000 and dbytes > 25000)
            or (proto in {"icmp", "arp"} and dur < 0.3)
        )
        label = "attack" if suspicious else "normal"

        records.append(
            {
                "dur": dur,
                "proto": proto,
                "service": service,
                "state": state,
                "spkts": spkts,
                "dpkts": dpkts,
                "sbytes": sbytes,
                "dbytes": dbytes,
                "sttl": int(rng.integers(20, 65)),
                "dttl": int(rng.integers(10, 90)),
                "rate": float(rng.uniform(0.1, 500.0)),
                "label": label,
            }
        )

    df = pd.DataFrame(records)
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_PATH, index=False)
    return df


def prepare_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Clean the dataset and create a model-ready feature table."""
    data = df.copy()

    if "label" in data.columns:
        label_mapping = {"normal": 0, "attack": 1, "NORMAL": 0, "ATTACK": 1}
        data["label"] = data["label"].map(label_mapping).fillna(1).astype(int)
    else:
        raise ValueError("The dataset must include a label column before training.")

    for column in data.columns:
        if column == "label":
            continue

        if pd.api.types.is_numeric_dtype(data[column]):
            data[column] = pd.to_numeric(data[column], errors="coerce")
            median_value = data[column].median()
            data[column] = data[column].fillna(median_value)
        else:
            data[column] = data[column].fillna("unknown").astype(str)

    X = data.drop(columns=["label"]).copy()
    categorical_columns = [
        column for column in X.columns if not pd.api.types.is_numeric_dtype(X[column])
    ]
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
    y = data["label"].astype(int)

    return X, y


def build_model() -> RandomForestClassifier:
    """Create a simple random-forest classifier for traffic classification."""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
    )


def train_model(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    """Train the model on the prepared data and return it."""
    model = build_model()
    model.fit(X, y)
    return model


def split_dataset(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return train/test splits for model evaluation."""
    X, y = prepare_training_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test
