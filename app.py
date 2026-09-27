from __future__ import annotations

from io import BytesIO

import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.network_intrusion_detection.pipeline import (
    apply_detection_rules,
    find_real_dataset_path,
    prepare_attack_category_data,
    prepare_feature_data,
    prepare_training_data,
    read_traffic_dataset,
    train_model,
)


st.set_page_config(page_title="Network Security Analyzer", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner="Training traffic model...")
def cached_train_model(features: pd.DataFrame, labels: pd.Series):
    return train_model(features, labels)


@st.cache_resource(show_spinner="Training attack-category model...")
def cached_category_model(features: pd.DataFrame, labels: pd.Series):
    return train_model(features, labels)


st.title("🛡️ Network Security Analyzer")
st.caption("Traffic triage, model evaluation, and explainable rule alerts")

real_dataset = find_real_dataset_path(split="train")
real_test_dataset = find_real_dataset_path(split="test")
default_df = read_traffic_dataset(real_dataset) if real_dataset is not None else None

with st.sidebar:
    st.header("Data source")
    if real_dataset is None:
        st.caption("No local UNSW-NB15 training split found. Upload a labeled training CSV below.")
    else:
        st.caption(f"Training split: {real_dataset.name}")
    uploaded_file = st.file_uploader("Upload training CSV", type=["csv"])
    uploaded_test_file = st.file_uploader("Optional held-out testing CSV", type=["csv"])
    st.caption("Testing records are kept separate from model training.")

if uploaded_file is not None:
    raw_df = pd.read_csv(BytesIO(uploaded_file.getvalue()))
    raw_df.columns = raw_df.columns.astype(str).str.strip().str.lower()
    dataset_name = uploaded_file.name
    if "label" not in raw_df.columns:
        if default_df is None:
            st.error("An unlabeled CSV can only be scored after a labeled UNSW-NB15 training dataset is available.")
            st.stop()
        st.info("Uploaded CSV has no label column. The configured real training dataset is used to score these rows.")
        scoring_df = raw_df.copy()
        raw_df = default_df
        dataset_name = real_dataset.name if real_dataset is not None else "Generated demo data"
    else:
        scoring_df = raw_df
else:
    if default_df is None:
        st.error("Upload the labeled UNSW-NB15 training CSV to continue. A separate labeled testing CSV is optional.")
        st.stop()
    raw_df = default_df
    scoring_df = raw_df
    dataset_name = real_dataset.name

if uploaded_test_file is not None:
    evaluation_df = pd.read_csv(BytesIO(uploaded_test_file.getvalue()))
    evaluation_df.columns = evaluation_df.columns.astype(str).str.strip().str.lower()
    evaluation_name = uploaded_test_file.name
elif real_test_dataset is not None:
    evaluation_df = read_traffic_dataset(real_test_dataset)
    evaluation_name = real_test_dataset.name
else:
    evaluation_df = None
    evaluation_name = ""

if evaluation_df is not None and "label" not in evaluation_df.columns:
    st.error("The held-out testing CSV must include a label or Label column.")
    st.stop()

if uploaded_file is None or "label" in raw_df.columns:
    if evaluation_df is not None:
        scoring_df = evaluation_df

if "label" not in raw_df.columns:
    st.error("The training CSV must include a label or Label column.")
    st.stop()

try:
    X, y = prepare_training_data(raw_df)
except (KeyError, ValueError) as error:
    st.error(f"Could not prepare this dataset: {error}")
    st.stop()

if len(y.unique()) < 2 or len(y) < 10:
    st.error("Training and evaluation need at least 10 rows and both normal and attack labels.")
    st.stop()

if evaluation_df is not None:
    try:
        X_train, y_train = X, y
        X_test, y_test = prepare_training_data(evaluation_df)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        evaluation_caption = f"Evaluation: held-out dataset {evaluation_name}"
    except (KeyError, ValueError) as error:
        st.error(f"Could not prepare the held-out dataset: {error}")
        st.stop()
else:
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y if y.value_counts().min() >= 2 else None,
        )
        evaluation_caption = f"Evaluation: stratified holdout ({len(y_test):,} rows)"
    except ValueError as error:
        st.error(f"Could not create a train/test split: {error}")
        st.stop()

model = cached_train_model(X_train, y_train)
y_pred = model.predict(X_test)
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1 score": f1_score(y_test, y_pred, zero_division=0),
}

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Traffic Records", len(raw_df))
with col2:
    st.metric("Normal", int((y == 0).sum()))
with col3:
    st.metric("Attacks", int((y == 1).sum()))

st.caption(f"Training data: {dataset_name} · {evaluation_caption} · {len(y_test):,} evaluation rows")

overview_tab, evaluation_tab, inspect_tab = st.tabs(["Traffic overview", "Model evaluation", "Inspect traffic"])

with overview_tab:
    left_col, right_col = st.columns(2)
    with left_col:
        if "proto" in raw_df.columns:
            st.subheader("Protocols")
            st.bar_chart(raw_df["proto"].fillna("unknown").value_counts().head(12))
        else:
            st.info("No proto column was found in this CSV.")
    with right_col:
        if "state" in raw_df.columns:
            st.subheader("Connection states")
            st.bar_chart(raw_df["state"].fillna("unknown").value_counts().head(12))
        else:
            st.info("No state column was found in this CSV.")

    if "attack_cat" in raw_df.columns:
        categories = raw_df["attack_cat"].fillna("Unknown").astype(str).str.strip()
        categories = categories[categories.str.lower().isin({"", "-", "normal", "none", "nan"}) == False]
        if not categories.empty:
            st.subheader("Attack categories in dataset")
            st.bar_chart(categories.value_counts().head(12))

with evaluation_tab:
    st.subheader("Held-out performance")
    metric_columns = st.columns(4)
    for column, (name, value) in zip(metric_columns, metrics.items()):
        column.metric(name, f"{value:.1%}")

    st.subheader("Confusion matrix")
    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    st.dataframe(
        pd.DataFrame(matrix, index=["Actual normal", "Actual attack"], columns=["Predicted normal", "Predicted attack"]),
        use_container_width=True,
    )

    st.subheader("Most influential model features")
    importance = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(15)
    st.bar_chart(importance)
    st.caption("Feature importance is a global model summary, not proof that a feature caused a particular alert.")

with inspect_tab:
    st.subheader("Score a traffic record")
    if scoring_df.empty:
        st.warning("There are no uploaded records to score.")
        st.stop()
    row_index = st.number_input("Row number", min_value=0, max_value=len(scoring_df) - 1, value=0, step=1)
    sample = scoring_df.iloc[[int(row_index)]].copy()
    sample_features = prepare_feature_data(sample, excluded_columns={"label", "attack_cat"})
    sample_features = sample_features.reindex(columns=X_train.columns, fill_value=0)
    prediction = int(model.predict(sample_features)[0])
    confidence = float(model.predict_proba(sample_features)[0].max())

    if prediction == 1:
        st.error(f"ATTACK INDICATED · model confidence {confidence:.1%}")
    else:
        st.success(f"NORMAL INDICATED · model confidence {confidence:.1%}")

    st.write("**Rule-based signals**")
    rule_reasons = apply_detection_rules(sample.iloc[0])
    if rule_reasons:
        for reason in rule_reasons:
            st.warning(reason)
    else:
        st.write("No configured packet, byte-volume, or traffic-rate threshold was exceeded.")

    st.write("**Why the model flagged this row**")
    active_features = sample_features.iloc[0]
    feature_importance = pd.Series(model.feature_importances_, index=X_train.columns)
    active = feature_importance[active_features > 0].sort_values(ascending=False).head(5)
    if prediction == 1 and not active.empty:
        explanations = pd.DataFrame(
            {
                "Feature": active.index,
                "Row value": active_features[active.index].values,
                "Global importance": active.values,
            }
        )
        st.dataframe(explanations, hide_index=True, use_container_width=True)
        st.caption("These are influential features present in the row; they are indicators, not causal explanations.")
    elif prediction == 1:
        st.write("The model combined the encoded traffic features to indicate an attack; no single active feature stood out.")
    else:
        st.write("The model did not find a strong attack pattern in this row's features.")

    if prediction == 1 and "attack_cat" in raw_df.columns:
        try:
            category_X, category_y = prepare_attack_category_data(raw_df)
            if category_y.nunique() >= 2 and len(category_y) >= 10:
                category_model = cached_category_model(category_X, category_y)
                category_sample = prepare_feature_data(sample, excluded_columns={"label", "attack_cat"})
                category_sample = category_sample.reindex(columns=category_X.columns, fill_value=0)
                category_prediction = category_model.predict(category_sample)[0]
                st.info(f"Likely attack category: **{category_prediction}**")
            else:
                st.info("Attack categories are present, but there are not enough labeled attack examples across multiple categories to train this classifier.")
        except ValueError as error:
            st.info(f"Attack-category prediction is unavailable: {error}")

    st.write("**Traffic record**")
    st.dataframe(sample, hide_index=True, use_container_width=True)
