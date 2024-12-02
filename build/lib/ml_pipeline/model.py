import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, f1_score, recall_score, precision_score
from ml_pipeline.preprocessing import preprocess_data
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

def train_model(df, target_column):
    st.sidebar.title("Model Training")

    test_size = st.sidebar.slider("Test Size (Fraction)", 0.1, 0.5, 0.2)

    # Split data
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Apply preprocessing after the split
    X_train = preprocess_data(X_train, scaling=True, encoding=True, imputation_strategy="mean", target_column=target_column)
    X_test = preprocess_data(X_test, scaling=True, encoding=True, imputation_strategy="mean", target_column=target_column)


    

    
    # Classification metrics
    if len(y.unique()) == 2:  # If it's a binary classification
        model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "XGBoost"], key="model_selection")
            # Train the selected model
        if model_choice == "Random Forest":
            model = RandomForestClassifier(random_state=42)
        elif model_choice == "XGBoost":
            model = XGBClassifier(random_state=42)

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Evaluate the model
        st.write("Model Evaluation:")
        st.write(f"Accuracy: {model.score(X_test, y_test):.4f}")
        st.write(f"F1 Score: {f1_score(y_test, predictions):.4f}")
        st.write(f"Recall Score: {recall_score(y_test, predictions):.4f}")
        st.write(f"Precision Score: {precision_score(y_test, predictions):.4f}")

        # Plot predictions
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=predictions)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs Predicted")
        st.pyplot(plt)

    else:
        #Regression
        model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "XGBoost"], key="model_selection")
            # Train the selected model
        if model_choice == "Random Forest":
            model = RandomForestRegressor(random_state=42)
        elif model_choice == "XGBoost":
            model = XGBRegressor(random_state=42)
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        # Evaluate the model
        st.write("Model Evaluation:")
        st.write(f"Accuracy: {model.score(X_test, y_test):.4f}")
        st.write(f"r2 Score: {r2_score(y_test, predictions):.4f}")