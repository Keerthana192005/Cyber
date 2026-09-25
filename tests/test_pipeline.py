from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from network_intrusion_detection.pipeline import prepare_training_data, load_or_create_demo_dataset, train_model


def test_load_or_create_demo_dataset_returns_expected_columns():
    df = load_or_create_demo_dataset(n_rows=50)

    assert isinstance(df, pd.DataFrame)
    assert {"dur", "sbytes", "dbytes", "proto", "state", "label"}.issubset(df.columns)
    assert len(df) == 50


def test_prepare_training_data_returns_features_and_labels():
    df = load_or_create_demo_dataset(n_rows=120)
    X, y = prepare_training_data(df)

    assert list(X.columns) == sorted(X.columns.tolist()) or len(X.columns) > 0
    assert set(y.unique()).issubset({0, 1})
    assert len(X) == len(y)


def test_train_model_accepts_preprocessed_features():
    df = load_or_create_demo_dataset(n_rows=200)
    X, y = prepare_training_data(df)

    model = train_model(X, y)

    assert model.n_features_in_ == X.shape[1]
    assert len(model.classes_) == 2
