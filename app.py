from __future__ import annotations

import streamlit as st

from src.network_intrusion_detection.pipeline import load_or_create_demo_dataset, prepare_training_data, train_model


st.set_page_config(page_title="Network Security Analyzer", page_icon="🛡️", layout="wide")

st.title("🛡️ Network Security Analyzer")
st.caption("A simple machine-learning demo for detecting suspicious network traffic.")

raw_df = load_or_create_demo_dataset(n_rows=2000)
X, y = prepare_training_data(raw_df)
model = train_model(X, y)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Records", len(raw_df))
with col2:
    st.metric("Normal Traffic", int((y == 0).sum()))
with col3:
    st.metric("Suspicious Traffic", int((y == 1).sum()))

st.subheader("Traffic overview")

protocol_counts = raw_df["proto"].value_counts().head(10)
state_counts = raw_df["state"].value_counts().head(10)

left_col, right_col = st.columns(2)
with left_col:
    st.bar_chart(protocol_counts)
with right_col:
    st.bar_chart(state_counts)

st.subheader("Predictive result")

sample = raw_df.sample(1).copy()
feature_columns = X.columns
sample_X, _ = prepare_training_data(sample)
sample_X = sample_X.reindex(columns=feature_columns, fill_value=0)

prediction = model.predict(sample_X)[0]
confidence = model.predict_proba(sample_X)[0].max()

status = "SUSPICIOUS" if prediction == 1 else "NORMAL"
color = "#ff4b4b" if prediction == 1 else "#00c853"

st.markdown(
    f"<div style='padding: 1rem; border-radius: 0.75rem; background: {color}; color: white; font-size: 2rem; text-align: center;'>"
    f"{status} · confidence: {confidence:.2%}"
    f"</div>",
    unsafe_allow_html=True,
)

st.write(sample)

st.subheader("What the model uses")

st.write("The model learns from traffic features such as connection duration, protocol, packet counts, and bytes transferred to decide whether a session looks normal or suspicious.")
