# Network Intrusion Detection & Traffic Analyzer

This project is a beginner-friendly Python-based network intrusion detection workflow built around the idea of analyzing traffic patterns to decide whether a connection looks normal or suspicious.

## Project story

The system follows a practical cybersecurity workflow:

1. Load traffic records
2. Clean and understand the data
3. Transform important features
4. Train a machine learning classifier
5. Show the final decision in a simple Streamlit dashboard

The goal is to make the project easy to explain in interviews and simple to extend in a real lab environment.

## Core idea

A network connection can be described by values such as:

- protocol
- packet counts
- byte counts
- duration
- connection state
- transfer rate

These values form the traffic fingerprint. A model can learn from examples of normal and attack traffic and then classify new traffic records.

## Tech stack

- Python
- Pandas
- NumPy
- scikit-learn
- Streamlit
- Matplotlib

## Dashboard capabilities

- Load local UNSW-NB15 training/testing CSV or Parquet splits automatically, or upload labeled training/testing CSVs.
- Upload an unlabeled CSV to score its traffic records after a labeled training dataset is configured.
- Review accuracy, precision, recall, F1 score, and the confusion matrix.
- Inspect global feature importance and the features present in an individual flagged record.
- Apply readable packet-count, byte-volume, and traffic-rate threshold rules.
- Predict an attack category when the training CSV includes enough labeled `attack_cat` examples.

## Real dataset setup

Place the converted files in `data/` using these names:

```text
data/UNSW_NB15_training-set.csv
data/UNSW_NB15_testing-set.csv
```

The app trains only on the training split and evaluates on the separate testing split when present. If no test split is supplied, it creates a stratified holdout from the training data. Dataset files are excluded from Git; for Streamlit Cloud, upload the labeled training CSV and optional testing CSV in the sidebar.

The converted training split contains 175,341 rows and 36 columns; the test split contains 82,332 rows and 36 columns. The demo CSV had 50 rows and 12 columns. Both formats share `dur`, `proto`, `service`, `state`, `spkts`, `dpkts`, `sbytes`, `dbytes`, `rate`, and `label`. The demo-only `sttl` and `dttl` fields are absent from these real split files, so they are not fabricated or required.

The real files add traffic-load (`sload`, `dload`), loss (`sloss`, `dloss`), inter-packet timing/jitter (`sinpkt`, `dinpkt`, `sjit`, `djit`), TCP windows/sequence/handshake timing (`swin`, `dwin`, `stcpb`, `dtcpb`, `tcprtt`, `synack`, `ackdat`), packet means (`smean`, `dmean`), HTTP/FTP indicators and counters, flow-count features, and `attack_cat`. Numeric and categorical traffic predictors are retained and encoded. `label` is the binary target; `attack_cat` is held out of binary-model features to prevent target leakage and is used separately for attack-category training.

The bounded Random Forest was trained on the real training split and evaluated on the untouched real testing split: accuracy 84.62%, precision 78.86%, recall 98.45%, and F1 87.57% in the current environment. Rule thresholds are illustrative heuristics, and model feature importance is not a causal explanation.

## Directory structure

- `src/network_intrusion_detection/pipeline.py` – dataset loader, preprocessing, and model functions
- `app.py` – Streamlit dashboard
- `tests/test_pipeline.py` – verification tests

## Quick start

```bash
python -m pip install -r requirements.txt
python -m pytest
streamlit run app.py
```

## Example explanation for interviews

> I built a Python-based network intrusion detection project using traffic records to identify whether communication appears normal or suspicious. The pipeline involved cleaning the dataset, selecting and engineering useful connection features, training a machine learning classifier, and comparing the predictions with traffic patterns such as packet flow, duration, and byte transfer. I then packaged the result into a dashboard to visualize the traffic distribution and model predictions.

## Notes

The application does not silently substitute synthetic demo traffic for real records. A labeled real training CSV is required to use the dashboard.
