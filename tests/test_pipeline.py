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


def test_real_unsw_style_csv_is_supported(tmp_path):
    dataset_path = tmp_path / "UNSW_NB15_training.csv"
    pd.DataFrame(
        {
            "srcip": ["10.0.0.1", "10.0.0.2"],
            "dstip": ["10.0.0.3", "10.0.0.4"],
            "proto": ["tcp", "udp"],
            "service": ["http", "dns"],
            "state": ["FIN", "INT"],
            "dur": [0.2, 1.4],
            "sbytes": [1000, 2200],
            "dbytes": [900, 1800],
            "spkts": [20, 55],
            "dpkts": [15, 35],
            "label": ["normal", "attack"],
        }
    ).to_csv(dataset_path, index=False)

    df = load_or_create_demo_dataset(dataset_path=dataset_path)
    X, y = prepare_training_data(df)

    assert len(df) == 2
    assert set(y.unique()).issubset({0, 1})
    assert X.shape[0] == 2
