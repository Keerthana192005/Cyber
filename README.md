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

This repository uses a generated demo dataset to keep the project runnable without downloading the large UNSW-NB15 file. It is designed as a clean starter for learning and demonstration.
