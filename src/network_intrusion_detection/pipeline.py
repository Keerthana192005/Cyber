from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNSW_TRAIN_DATASET_CANDIDATES = [
    PROJECT_ROOT / "data" / "UNSW_NB15_training-set.csv",
    PROJECT_ROOT / "data" / "UNSW_NB15_training-set.parquet",
    PROJECT_ROOT / "UNSW_NB15_training-set.csv",
    PROJECT_ROOT / "UNSW_NB15_training-set.parquet",
]
UNSW_TEST_DATASET_CANDIDATES = [
    PROJECT_ROOT / "data" / "UNSW_NB15_testing-set.csv",
    PROJECT_ROOT / "data" / "UNSW_NB15_testing-set.parquet",
    PROJECT_ROOT / "UNSW_NB15_testing-set.csv",
    PROJECT_ROOT / "UNSW_NB15_testing-set.parquet",
]


def _normalize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns commonly seen in UNSW-NB15 or similar traffic datasets."""
    renamed_columns = {
        "attack_cat": "attack_cat",
        "Label": "label",
        "label": "label",
        "dur": "dur",
        "proto": "proto",
        "service": "service",
        "state": "state",
        "spkts": "spkts",
        "dpkts": "dpkts",
        "sbytes": "sbytes",
        "dbytes": "dbytes",
        "sttl": "sttl",
        "dttl": "dttl",
        "rate": "rate",
        "srcip": "srcip",
        "dstip": "dstip",
        "sport": "sport",
        "dsport": "dsport",
    }
    df = df.rename(columns={k: v for k, v in renamed_columns.items() if k in df.columns})
    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.strip().str.lower()
    return df


def find_real_dataset_path(split: str = "train") -> Path | None:
    """Find the requested local UNSW-NB15 training or testing split."""
    if split not in {"train", "test"}:
        raise ValueError("split must be either 'train' or 'test'.")

    candidates = UNSW_TRAIN_DATASET_CANDIDATES if split == "train" else UNSW_TEST_DATASET_CANDIDATES
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def read_traffic_dataset(dataset_path: Path | str) -> pd.DataFrame:
    """Read a CSV or Parquet traffic dataset and normalize its label column."""
    path = Path(dataset_path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif path.suffix.lower() in {".csv", ".txt"}:
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix}")
    return _normalize_feature_names(df)


def convert_parquet_to_csv(parquet_path: Path | str, csv_path: Path | str | None = None) -> Path:
    """Convert one Parquet dataset to CSV, preserving its columns and row order."""
    source_path = Path(parquet_path)
    if source_path.suffix.lower() not in {".parquet", ".pq"}:
        raise ValueError("The source dataset must be a Parquet file.")
    destination_path = Path(csv_path) if csv_path is not None else source_path.with_suffix(".csv")
    pd.read_parquet(source_path).to_csv(destination_path, index=False)
    return destination_path


def load_or_create_demo_dataset(n_rows: int = 2000, dataset_path: Path | str | None = None) -> pd.DataFrame:
    """Load a real UNSW-NB15 training split or create reproducible demo traffic."""
    path = Path(dataset_path) if dataset_path is not None else find_real_dataset_path()

    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        return read_traffic_dataset(path)

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

    return _normalize_feature_names(pd.DataFrame(records))


def prepare_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Clean the dataset and create a model-ready feature table."""
    data = df.copy()

    if "label" not in data.columns:
        raise ValueError("The dataset must include a label column before training.")

    label_mapping = {
        "normal": 0,
        "attack": 1,
        "0": 0,
        "1": 1,
        "0.0": 0,
        "1.0": 1,
        "abnormal": 1,
        "benign": 0,
        "anomaly": 1,
        "dos": 1,
        "exploits": 1,
        "fuzzers": 1,
        "generic": 1,
        "reconnaissance": 1,
        "analysis": 1,
        "backdoor": 1,
        "shellcode": 1,
        "worms": 1,
        "normal.0": 0,
        "attack.0": 1,
    }
    data["label"] = data["label"].astype(str).str.strip().str.lower().map(label_mapping)
    if data["label"].isna().any():
        data["label"] = data["label"].fillna(1)
    data["label"] = data["label"].astype(int)

    X = prepare_feature_data(data, excluded_columns={"label", "attack_cat"})
    y = data["label"].astype(int)

    return X, y


def prepare_feature_data(
    df: pd.DataFrame, excluded_columns: set[str] | None = None
) -> pd.DataFrame:
    """Convert traffic records into numeric model features."""
    excluded = set(excluded_columns or ()) | {"srcip", "dstip", "id"}
    X = df.drop(columns=[column for column in excluded if column in df.columns]).copy()

    for column in ("sport", "dsport"):
        if column in X.columns:
            X[column] = pd.to_numeric(X[column], errors="coerce")

    for column in X.columns:
        if pd.api.types.is_numeric_dtype(X[column]):
            X[column] = pd.to_numeric(X[column], errors="coerce")
            median_value = X[column].median()
            X[column] = X[column].fillna(0 if pd.isna(median_value) else median_value)
        else:
            X[column] = X[column].fillna("unknown").astype(str)

    categorical_columns = [column for column in X.columns if not pd.api.types.is_numeric_dtype(X[column])]
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True, dtype=int)
    return X


def prepare_attack_category_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return numeric features and attack-category labels for attack records."""
    if "attack_cat" not in df.columns:
        raise ValueError("The dataset does not contain an attack_cat column.")

    categories = df["attack_cat"].fillna("").astype(str).str.strip()
    attack_mask = ~categories.str.lower().isin({"", "-", "normal", "none", "nan"})
    attack_rows = df.loc[attack_mask].copy()
    y = categories.loc[attack_mask].reset_index(drop=True)
    X = prepare_feature_data(
        attack_rows,
        excluded_columns={"label", "attack_cat"},
    ).reset_index(drop=True)
    return X, y


def apply_detection_rules(record: pd.Series | dict) -> list[str]:
    """Return human-readable reasons when simple traffic thresholds are exceeded."""
    values = record if isinstance(record, dict) else record.to_dict()
    rules = (
        ("spkts", 350, "Source packet count is unusually high (spkts > 350)."),
        ("dpkts", 400, "Destination packet count is unusually high (dpkts > 400)."),
        ("sbytes", 30000, "Source byte volume is unusually high (sbytes > 30,000)."),
        ("dbytes", 40000, "Destination byte volume is unusually high (dbytes > 40,000)."),
        ("rate", 300, "Traffic rate is unusually high (rate > 300)."),
    )
    triggered = []
    for column, threshold, message in rules:
        value = pd.to_numeric(values.get(column), errors="coerce")
        if pd.notna(value) and value > threshold:
            triggered.append(message)
    return triggered


def build_model() -> RandomForestClassifier:
    """Create a bounded random-forest classifier for traffic classification."""
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
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
