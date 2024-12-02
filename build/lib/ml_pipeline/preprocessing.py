import streamlit as st
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
import re

import re

def process_column_names(df):
    """ Clean the column names by replacing non-alphanumeric characters (except spaces) with underscores.
    """
    # Replace all non-alphanumeric characters except spaces with underscores
    df.columns = [re.sub(r'[^\w\s]', '_', col) for col in df.columns]
    
    # Replace multiple consecutive underscores with a single underscore
    df.columns = [re.sub(r'_{2,}', '_', col) for col in df.columns]

    return df



def convert_categorical(df):
    """
    The input dataframe contains only numeric columns
    """
    label_encoder = LabelEncoder()
    num_cols = df.select_dtypes(["int","float"]).columns
    for col in num_cols:
        num_unique_values = len(df[col].unique())  # Count unique values
        
        if num_unique_values < 10:
            # If fewer than 10 unique values, apply Label Encoding
            df[col] = label_encoder.fit_transform(df[col])
    
    return df
def preprocessing_insights(df):
    st.title("Preprocessing Insights")
    
    # Missing values insights
    st.subheader("Missing Values")
    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()
    if total_missing > 0:
        st.write(f"Your dataset contains {total_missing} missing values across {len(missing_values[missing_values > 0])} columns.")
        st.markdown("- **Recommendation:** Impute missing values with mean/median for numeric data or mode for categorical data.")
        st.markdown("- Alternatively, drop rows/columns with excessive missing values.")
    else:
        st.write("No missing values detected.")
    
    # Categorical encoding insights
    st.subheader("Categorical Variables")
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(categorical_cols) > 0:
        st.write(f"The dataset contains {len(categorical_cols)} categorical columns: {', '.join(categorical_cols)}.")
        st.markdown("- **Recommendation:** Use one-hot encoding, label encoding, or target encoding for these columns based on model requirements.")
    else:
        st.write("No categorical variables detected.")
    
    # Scaling insights
    st.subheader("Numeric Scaling")
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    if len(numeric_cols) > 0:
        st.write(f"The dataset contains {len(numeric_cols)} numeric columns.")
        st.markdown("- **Recommendation:** Apply scaling (e.g., StandardScaler, MinMaxScaler) to these columns to normalize data distribution.")
    else:
        st.write("No numeric variables detected.")
    
    # Outlier handling insights
    st.subheader("Outliers")
    st.write("Outlier detection was performed. Review the outlier insights from the EDA section.")
    st.markdown("- **Recommendation:** Consider removing or capping extreme outliers to prevent model distortion.")



def preprocess_data(df, scaling, encoding, imputation_strategy,target_column):
    # Separate numeric and categorical columns
    numeric_columns = df.select_dtypes(include=["int", "float"]).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns
    if target_column in numeric_columns:
        numeric_columns.remove(target_column)
    # Impute numeric data
    imputer = SimpleImputer(strategy=imputation_strategy)
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    
    # Impute categorical data (using most_frequent strategy)
    if categorical_columns.size > 0:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df[categorical_columns] = cat_imputer.fit_transform(df[categorical_columns])

    # Scaling Numeric Data
    if scaling:
        scaler = StandardScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    # Encoding Categorical Columns
    if encoding:
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    return process_column_names(df)


def preprocessing_options(df, target_column):
    st.sidebar.title("Preprocessing Options")
    
    scaling = st.sidebar.checkbox("Scale Numeric Data", value=True)
    encoding = st.sidebar.checkbox("Encode Categorical Data", value=True)
    imputation_strategy = st.sidebar.selectbox(
        "Missing Value Imputation Strategy", ["mean", "median", "most_frequent"], key="Imputation"
    )
    
    # Select categorical and numeric columns
    df_categorical = df.select_dtypes(["O"])
    df_numeric = df.select_dtypes(exclude=["O"])

    # Apply transformation to categorical columns (if needed)
    df_numeric = convert_categorical(df_numeric)

    # Concatenate transformed categorical columns with numeric columns
    df_transformed = pd.concat([df_numeric, df_categorical], axis=1)

    # Apply preprocessing
    processed_df = preprocess_data(df_transformed, scaling, encoding, imputation_strategy, target_column)
    st.write("Preprocessed Data Preview:")
    st.dataframe(processed_df.head())
    return processed_df

