import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from ml_pipeline.eda import *
from ml_pipeline.preprocessing import *
from ml_pipeline.model import train_model

def main():
    st.header("Exploratory Data Analysis")


    # File uploader for any file type
    uploaded_file = st.file_uploader("Upload a data file (CSV, Excel, JSON, etc.)", type=None)

    if uploaded_file:
        # Determine file type by extension
        file_name = uploaded_file.name
        file_extension = file_name.split('.')[-1].lower()

        # Handle CSV files
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        # Handle Excel files
        elif file_extension in ['xls', 'xlsx']:
            df = pd.read_excel(uploaded_file)
        # Handle JSON files
        elif file_extension == 'json':
            df = pd.read_json(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV, Excel, or JSON file.")
            st.stop()  # Stops execution if the file format is unsupported

        # Display data preview
        st.write("Data Preview:")
        st.dataframe(df.head())

        # Streamlit App
        data_overview(df)

        # Check for missing values
        check_missing_values(df)

        # Display data types and unique values
        data_types_and_uniques(df)

        # Handle duplicate rows
        handle_duplicates(df)

        # Univariate analysis
        st.title("Univariate Analysis")
        univariate_analysis(df)

        # Bivariate analysis
        st.title("Bivariate Analysis")
        target_column = st.selectbox("Select Target Column", df.columns, key="Bivariate Analysis")
        
        # Perform bivariate analysis if a target column is selected
        if target_column:
            st.write(f"Performing bivariate analysis with target column: {target_column}")
            bivariate_analysis(df, target_column)

        st.title("Categorical Count Plot Analysis")
        visualize_categorical(df)

        # Generate Correlation Heatmap
        correlation_heatmap(df)

        # Outlier Detection for numeric columns
        numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns
        detect_outliers(df, numeric_columns)

        # Preprocessing Summary
        preprocessing_insights(df)
        
        # Model Training
        st.title("Model Training")
        target_column = st.selectbox("Select Target Column", df.columns, key="model training")
        
        if target_column:
            processed_df = preprocessing_options(df, target_column)
            train_model(processed_df, target_column)

# If this script is executed, the main() function will be called
if __name__ == "__main__":
    main()