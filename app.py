import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Dataset Poisoning Tool", layout="wide")

st.sidebar.title("Machine Learning Project")
st.sidebar.markdown("### Dataset Poisoning Detector")

menu = st.sidebar.radio("Navigation", ["1. Upload & EDA", "2. Model Training", "3. Poisoning Analysis"])

st.sidebar.markdown("---")
st.sidebar.write("Developed by: ML Student")

# Helper to load data
def load_data():
    if os.path.exists("dataset.csv"):
        return pd.read_csv("dataset.csv")
    return None

df = load_data()

if menu == "1. Upload & EDA":
    st.title("Data Upload & Exploratory Data Analysis")
    st.write("Welcome to the Data Analyzer! Here we analyze the dataset before training.")
    
    if df is not None:
        st.success("Default dataset (dataset.csv) loaded successfully.")
        
        st.subheader("Data Preview")
        st.dataframe(df.head(10))
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dataset Shape")
            st.write(f"Rows: {df.shape[0]}")
            st.write(f"Columns: {df.shape[1]}")
            
        with col2:
            st.subheader("Missing Values")
            st.write(df.isnull().sum())
            
        st.subheader("Statistical Description")
        st.write(df.describe())
        
    else:
        st.error("Dataset not found! Please make sure dataset.csv exists in the directory.")

elif menu == "2. Model Training":
    st.title("Train Machine Learning Model")
    if df is not None:
        st.write("In this section, we will train a baseline Random Forest model on the clean dataset.")
        
        X = df.drop(columns=['target'])
        y = df['target']
        
        test_size = st.slider("Test Size Percentage", 10, 40, 20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
        
        if st.button("Train Model"):
            with st.spinner("Training Random Forest Classifier..."):
                model = RandomForestClassifier(random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                
                st.success("Model trained successfully!")
                st.metric(label="Model Accuracy", value=f"{acc*100:.2f}%")
                
                # Show some predictions
                results_df = pd.DataFrame({"Actual": y_test, "Predicted": preds})
                st.write("Sample Predictions vs Actual:")
                st.dataframe(results_df.head(10))
    else:
        st.warning("Please load data in the EDA section first.")

elif menu == "3. Poisoning Analysis":
    st.title("Data Poisoning Simulation & Defense")
    if df is not None:
        st.write("Simulate an attacker poisoning your dataset, and use Anomaly Detection to defend against it.")
        
        poison_ratio = st.slider("Select Poisoning Ratio (%)", 5, 30, 15) / 100
        
        if st.button("Run Poisoning Experiment"):
            X = df.drop(columns=['target'])
            y = df['target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 1. Clean Baseline
            clean_model = RandomForestClassifier(random_state=42)
            clean_model.fit(X_train, y_train)
            clean_acc = accuracy_score(y_test, clean_model.predict(X_test))
            
            # 2. Poison Data
            st.info("Injecting poisoned samples into the training data...")
            X_train_pois = X_train.copy().values
            y_train_pois = y_train.copy().values
            n_samples = len(y_train)
            n_poison = int(n_samples * poison_ratio)
            
            np.random.seed(42)
            poison_idx = np.random.choice(n_samples, n_poison, replace=False)
            
            for i in poison_idx:
                # Random label flip
                y_train_pois[i] = (y_train_pois[i] + 1) % 3
                
            pois_model = RandomForestClassifier(random_state=42)
            pois_model.fit(X_train_pois, y_train_pois)
            pois_acc = accuracy_score(y_test, pois_model.predict(X_test))
            
            # 3. Defend using Isolation Forest
            st.info("Running Unsupervised Anomaly Detection to find malicious rows...")
            X_comb = np.column_stack((X_train_pois, y_train_pois))
            detector = IsolationForest(contamination=poison_ratio + 0.05, random_state=42)
            detector.fit(X_comb)
            
            preds = detector.predict(X_comb)
            suspicious_mask = (preds == -1)
            
            st.write(f"⚠️ **Identified {sum(suspicious_mask)} suspicious rows in the training data.**")
            
            # 4. Retrain
            clean_mask = ~suspicious_mask
            rec_model = RandomForestClassifier(random_state=42)
            rec_model.fit(X_train_pois[clean_mask], y_train_pois[clean_mask])
            rec_acc = accuracy_score(y_test, rec_model.predict(X_test))
            
            st.success("Experiment Completed! Check the results below.")
            
            # Show results
            st.subheader("Accuracy Comparison")
            col1, col2, col3 = st.columns(3)
            col1.metric("Clean Model", f"{clean_acc*100:.2f}%")
            col2.metric("Poisoned Model", f"{pois_acc*100:.2f}%", delta=f"{(pois_acc - clean_acc)*100:.2f}%")
            col3.metric("Recovered Model", f"{rec_acc*100:.2f}%", delta=f"{(rec_acc - pois_acc)*100:.2f}%")
            
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(['Clean Data', 'Poisoned Data', 'Recovered Data'], 
                   [clean_acc*100, pois_acc*100, rec_acc*100], 
                   color=['#2ca02c', '#d62728', '#1f77b4'])
            ax.set_ylabel("Accuracy (%)")
            ax.set_ylim(0, 100)
            for i, v in enumerate([clean_acc*100, pois_acc*100, rec_acc*100]):
                ax.text(i, v + 2, f"{v:.1f}%", ha='center')
            st.pyplot(fig)
