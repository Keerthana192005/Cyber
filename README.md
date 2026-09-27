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

- Upload a labeled CSV to train and evaluate on a held-out split, or upload an unlabeled CSV to score its traffic records.
- Review accuracy, precision, recall, F1 score, and the confusion matrix.
- Inspect global feature importance and the features present in an individual flagged record.
- Apply readable packet-count, byte-volume, and traffic-rate threshold rules.
- Predict an attack category when the training CSV includes enough labeled `attack_cat` examples.

The dashboard accepts common UNSW-NB15 fields, including `Label`, `attack_cat`, `proto`, `service`, `state`, `dur`, packet/byte counts, and ports. Source/destination IP addresses are excluded from model features to avoid high-cardinality identifiers.

The bundled dataset is generated demo data, not UNSW-NB15. Rule thresholds are illustrative heuristics and model feature importance is not a causal explanation. For credible evaluation, provide the real dataset and report metrics from its held-out records.

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

The app automatically uses a local UNSW-NB15 CSV if it finds one under `data/` or in the project root. Otherwise, it generates reproducible demo traffic in memory. Dataset CSV files are excluded from Git by `.gitignore`.
