import streamlit as st
import pandas as pd
from dataset_module import StrokeDataset
from eda_module import EDA
from ml_model_module import MLModel
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Stroke Data Analytics", layout="wide")

# --- Sidebar ---
st.sidebar.title("Stroke Data Analytics")
option = st.sidebar.selectbox(
    "Choose a step:",
    [
        "Load Dataset",
        "EDA & Data Cleaning",
        "Feature Engineering",
        "Prepare ML Data",
        "Train & Evaluate Models",
        "Visualize Performance"
    ]
)

# --- Initialize session state ---
if 'raw_df' not in st.session_state:
    st.session_state.raw_df = None
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'featured_df' not in st.session_state:
    st.session_state.featured_df = None
if 'eda_processor' not in st.session_state:
    st.session_state.eda_processor = None
if 'ml_splits' not in st.session_state:
    st.session_state.ml_splits = {}
if 'ml_models' not in st.session_state:
    st.session_state.ml_models = {}
if 'ml_results' not in st.session_state:
    st.session_state.ml_results = {}

ml_targets = ["Chronic Stress", "Physical Activity", "Income Level", "stroke"]

# --- Option: Load Dataset ---
if option == "Load Dataset":
    st.title("Load Dataset")
    uploaded_file = st.file_uploader("Upload your stroke dataset (.csv)", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.raw_df = df
        st.session_state.eda_processor = EDA(df.copy())
        st.write("### Preview of Uploaded Dataset")
        st.dataframe(df.head())

# --- Option: EDA & Data Cleaning ---
elif option == "EDA & Data Cleaning":
    st.title("EDA & Data Cleaning")
    if st.session_state.raw_df is not None:
        eda = st.session_state.eda_processor
        st.write("### Initial Dataset Info")
        st.write(st.session_state.raw_df.describe())

        st.write("### Handling Missing Data")
        st.session_state.cleaned_df = eda.clean_missing_data(st.session_state.raw_df.copy())
        st.write("### Cleaned Data Preview")
        st.dataframe(st.session_state.cleaned_df.head())
    else:
        st.warning("Please load a dataset first.")

# --- Option: Feature Engineering ---
elif option == "Feature Engineering":
    st.title("Feature Engineering")
    if st.session_state.cleaned_df is not None:
        eda = st.session_state.eda_processor
        st.session_state.featured_df = eda.create_features(st.session_state.cleaned_df.copy())
        st.write("### Feature Engineered Data")
        st.dataframe(st.session_state.featured_df.head())
    else:
        st.warning("Please perform EDA and cleaning first.")

# --- Option: Prepare ML Data ---
elif option == "Prepare ML Data":
    st.title("Prepare Data for ML")
    if st.session_state.featured_df is not None:
        eda = st.session_state.eda_processor
        st.session_state.ml_splits.clear()
        for target_col in ml_targets:
            X_train, X_test, y_train, y_test = eda.split_dataset(st.session_state.featured_df.copy(), target_col)
            if target_col == 'stroke':
                X_train, y_train = eda.address_class_imbalance(X_train, y_train, strategy='smote')
            st.session_state.ml_splits[target_col] = {
                'X_train': X_train, 'X_test': X_test,
                'y_train': y_train, 'y_test': y_test
            }
        st.success("Data prepared for ML models.")
    else:
        st.warning("Please complete feature engineering first.")

# --- Option: Train & Evaluate Models ---
elif option == "Train & Evaluate Models":
    st.title("Train and Evaluate Models")
    if st.session_state.ml_splits:
        st.session_state.ml_results.clear()
        st.session_state.ml_models.clear()
        for target, splits in st.session_state.ml_splits.items():
            ml_model = MLModel(
                splits['X_train'], splits['X_test'],
                splits['y_train'], splits['y_test']
            )
            ml_model.train_and_evaluate(target)
            st.session_state.ml_results[target] = ml_model.results[target]
            st.session_state.ml_models[target] = ml_model
        st.success("Models trained and evaluated.")
    else:
        st.warning("Please prepare data for ML first.")

# --- Option: Visualize Performance ---
elif option == "Visualize Performance":
    st.title("Model Performance Visualizations")

    # Display custom uploaded performance plots
    st.subheader("Performance for Chronic Stress")
    st.image("output.png", caption="Model Performance for Chronic Stress", use_container_width=True)
    
    st.subheader("Performance for Physical Activity")
    st.image("output_py.png", caption="Model Performance for Physical Activity", use_container_width=True)

    st.subheader("Performance for Income Level")
    st.image("output_in.png", caption="Model Performance for Income Level", use_container_width=True)
    
    st.subheader("Performance for Stroke")
    st.image("output_st.png", caption="Model Performance for Stroke", use_container_width=True)

    st.subheader("ROC")
    st.image("output_str_pre.png", caption="ROC", use_container_width=True)

    
    st.info("Note: These are static visualizations. For dynamic performance metrics or retraining, revisit earlier steps.")
    
    