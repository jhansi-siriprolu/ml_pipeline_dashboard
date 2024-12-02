import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
import warnings

warnings.filterwarnings("ignore")

def data_overview(df):
    """
    Display basic details about the dataset in Streamlit.
    """
    st.subheader("Dataset Overview")
    st.write("**Shape of the dataset:**", df.shape)
    st.write("**Dataset Info:**")
    buffer = df.info(buf=None)  # Capturing `info()` output requires redirecting `stdout`
    st.text(buffer)
    st.write("**Dataset Summary:**")
    st.dataframe(df.describe(include="all").transpose())

def check_missing_values(df):
    """
    Display missing values in the dataset in Streamlit.
    """
    st.subheader("Missing Values")
    missing_values = df.isnull().sum().sort_values(ascending=False)
    missing_percent = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing_df = pd.DataFrame({"Missing Count": missing_values, "Percentage (%)": missing_percent})
    missing_df = missing_df[missing_df["Missing Count"] > 0]
    if missing_df.empty:
        st.success("No missing values in the dataset.")
    else:
        st.write("**Missing Values Table:**")
        st.dataframe(missing_df)

def data_types_and_uniques(df):
    """
    Display data types and number of unique values in Streamlit.
    """
    st.subheader("Data Types and Unique Values")
    types_uniques = {col: {"Type": df[col].dtype, "Unique Values": df[col].nunique()} for col in df.columns}
    df_types_uniques = pd.DataFrame.from_dict(types_uniques, orient="index")
    # Sort Unique values in descending order
    df_types_uniques = df_types_uniques.sort_values(by="Unique Values", ascending=False)
    st.dataframe(df_types_uniques)

def handle_duplicates(df):
    """
    Drop duplicate rows and display the number of duplicates in Streamlit.
    """
    duplicate_count = df.duplicated().sum()
    st.subheader("Duplicate Rows")
    if duplicate_count > 0:
        st.warning(f"Number of duplicate rows: {duplicate_count}")
        if st.button("Remove Duplicates"):
            df.drop_duplicates(inplace=True)
            st.success("Duplicates removed successfully!")
    else:
        st.success("No duplicate rows found.")


# Univariate Analysis - Subplots in Grid
def univariate_analysis(df):
    """
    Perform univariate analysis on the dataset with subplots.
    """
    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns
    if len(numeric_columns) == 0:
        st.warning("No numeric columns found for univariate analysis.")
        return

    num_plots = len(numeric_columns)
    num_cols = 3  # You can adjust the number of columns in the grid
    num_rows = (num_plots // num_cols) + (num_plots % num_cols > 0)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))
    axes = axes.flatten()  # Flatten to make indexing easier
    
    for i, col in enumerate(numeric_columns):
        sns.histplot(df[col], kde=True, bins=30, ax=axes[i])
        axes[i].set_title(f"Distribution of {col}")
    
    # Remove empty subplots if there are any
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    st.pyplot(fig)

# Bivariate Analysis - Subplots in Grid
def bivariate_analysis(df, target_column):
    """
    Perform bivariate analysis between features and a target variable with subplots.
    """
    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns
    numeric_columns = [col for col in numeric_columns if col != target_column]

    if len(numeric_columns) == 0:
        st.warning("No numeric columns available for bivariate analysis.")
        return

    num_plots = len(numeric_columns)
    num_cols = 3
    num_rows = (num_plots // num_cols) + (num_plots % num_cols > 0)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_columns):
        sns.scatterplot(data=df, x=col, y=target_column, ax=axes[i])
        axes[i].set_title(f"{col} vs {target_column}")
    
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    st.pyplot(fig)

# Categorical Analysis - Subplots in Grid
def visualize_categorical(df):
    """
    Visualize categorical variables using bar plots in a grid.
    """
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(categorical_cols) == 0:
        st.warning("No categorical columns found for analysis.")
        return

    num_plots = len(categorical_cols)
    num_cols = 3
    num_rows = (num_plots // num_cols) + (num_plots % num_cols > 0)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))
    axes = axes.flatten()

    for i, col in enumerate(categorical_cols):
        sns.countplot(data=df, x=col, order=df[col].value_counts().index, ax=axes[i])
        axes[i].set_title(f"Count Plot of {col}")
        axes[i].tick_params(axis='x', rotation=45)

    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    st.pyplot(fig)

# Function to plot the correlation heatmap
def correlation_heatmap(df):
    """
    Plot a heatmap of the correlations.
    """
    st.subheader("Correlation Heatmap")
    
    # Compute the correlation matrix
    corr_matrix = df.select_dtypes(["int","float"]).corr()
    
    # Create a figure with a size suitable for Streamlit display
    plt.figure(figsize=(12, 8))
    
    # Plot the heatmap
    sns.heatmap(corr_matrix, annot=True, fmt=".1f", cmap="coolwarm", vmin=-1, vmax=1)
    
    # Set the title
    plt.title("Correlation Heatmap")
    
    # Display the plot in Streamlit
    st.pyplot(plt)

# Function to detect and visualize outliers using boxplots
def detect_outliers(df, num_cols):
    """
    Detect and visualize outliers using boxplots.
    """
    st.subheader("Outlier Detection")

    if len(num_cols) == 0:
        st.warning("No numeric columns available for outlier detection.")
        return

    # Create a subplot grid based on the number of numeric columns
    num_plots = len(num_cols)
    num_cols_grid = 3
    num_rows_grid = (num_plots // num_cols_grid) + (num_plots % num_cols_grid > 0)

    fig, axes = plt.subplots(num_rows_grid, num_cols_grid, figsize=(15, 5*num_rows_grid))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        sns.boxplot(x=df[col], ax=axes[i])
        axes[i].set_title(f"Boxplot of {col}")
    
    # Remove empty subplots if any
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    # Display the plot in Streamlit
    st.pyplot(fig)