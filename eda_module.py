# eda_module.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE # Ensure imbalanced-learn is installed
import warnings
warnings.filterwarnings("ignore")


class EDA:
    def __init__(self, dataframe):
        """
        Initializes the EDA class with the raw DataFrame.
        """
        self.raw_data = dataframe
        self.cleaned_data = None
        self.processed_data_for_ml = None # Stores data after encoding for ML

    def perform_initial_exploration(self):
        """
        Performs initial data exploration, prints info, missing values,
        descriptive statistics including skewness and kurtosis.
        """
        print("\n--- Initial Data Exploration ---")
        print("Data Info:")
        self.raw_data.info()
        print("\nMissing Values:")
        print(self.raw_data.isnull().sum())
        print("\nDescriptive Statistics (Numerical):")
        # Select only numerical columns for describe, skew, kurt
        numerical_cols = self.raw_data.select_dtypes(include=np.number).columns
        if not numerical_cols.empty:
            print(self.raw_data[numerical_cols].describe())
            print("\nSkewness (Numerical):")
            print(self.raw_data[numerical_cols].skew())
            print("\nKurtosis (Numerical):")
            print(self.raw_data[numerical_cols].kurt())
        else:
            print("No numerical columns to describe, calculate skewness or kurtosis.")

        print("\nValue Counts (Categorical):")
        for col in self.raw_data.select_dtypes(include='object').columns:
            print(f"\n{col}:")
            print(self.raw_data[col].value_counts())

    def clean_missing_data(self, df):
        """
        Handles missing data, specifically for 'bmi' and potentially 'gender'.
        Returns a DataFrame with missing values handled.
        """
        print("\n--- Cleaning Missing Data ---")
        # Handle 'Other' gender if present (as it usually has only 1 record and can cause issues)
        if 'gender' in df.columns and 'Other' in df['gender'].unique():
            df = df[df['gender'] != 'Other'].copy() # Use .copy() to avoid SettingWithCopyWarning
            df.reset_index(drop=True, inplace=True)
            print("Removed 'Other' gender record.")

        # Impute 'bmi' with its mean
        if 'bmi' in df.columns:
            # Convert 'bmi' to numeric, coercing errors to NaN to handle non-numeric entries if any
            df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
            if df['bmi'].isnull().any():
                bmi_mean = df['bmi'].mean()
                df['bmi'].fillna(bmi_mean, inplace=True)
                print(f"Imputed missing 'bmi' values with mean: {bmi_mean:.2f}")

        # Impute other common categorical missing data with mode if present
        for col in ['work_type', 'smoking_status', 'Residence_type', 'ever_married']:
            if col in df.columns and df[col].isnull().any():
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                print(f"Imputed missing '{col}' values with mode: '{mode_val}'")

        # Drop any remaining rows with NaN if they exist after targeted imputation
        initial_rows = df.shape[0]
        df.dropna(inplace=True)
        rows_after_dropna = df.shape[0]
        if initial_rows > rows_after_dropna:
            print(f"Dropped {initial_rows - rows_after_dropna} rows with remaining missing values.")


        self.cleaned_data = df
        print("Missing data cleaning complete.")
        return df

    def create_features(self, df):
        """
        Engineers new features from existing ones.
        Returns a DataFrame with new features.
        """
        print("\n--- Feature Engineering ---")
        # Example: BMI Category
        if 'bmi' in df.columns:
            bins = [0, 18.5, 24.9, 29.9, float('inf')]
            labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
            # Ensure 'bmi' is numeric before cutting
            df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
            df['bmi_category'] = pd.cut(df['bmi'], bins=bins, labels=labels, right=False)
            print("Created 'bmi_category'.")

        # Example: Age Groups
        if 'age' in df.columns:
            age_bins = [0, 18, 45, 65, float('inf')]
            age_labels = ['Child/Adolescent', 'Young Adult', 'Middle-Aged', 'Senior']
            df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
            print("Created 'age_group'.")

        # Example: Glucose Level Categories
        if 'avg_glucose_level' in df.columns:
            glucose_bins = [0, 100, 126, float('inf')]
            glucose_labels = ['Normal', 'Pre-diabetic', 'Diabetic']
            df['glucose_category'] = pd.cut(df['avg_glucose_level'], bins=glucose_bins, labels=glucose_labels, right=False)
            print("Created 'glucose_category'.")

        # Interaction feature example
        if 'age' in df.columns and 'hypertension' in df.columns:
            df['age_hypertension_interaction'] = df['age'] * df['hypertension']
            print("Created 'age_hypertension_interaction'.")

        # --- New Feature: Smoking and Heart Disease Risk ---
        # Assuming higher risk if smokes and has heart disease
        if 'smoking_status' in df.columns and 'heart_disease' in df.columns:
            df['smoking_heart_risk'] = df.apply(
                lambda row: 1 if 'smokes' in str(row['smoking_status']).lower() and row['heart_disease'] == 1 else 0,
                axis=1
            )
            print("Created 'smoking_heart_risk' feature.")

        # --- New Feature: Activity and Income Interaction ---
        if 'Physical Activity' in df.columns and 'Income Level' in df.columns:
            df['activity_income_interaction'] = df.apply(
                lambda row: 1 if row['Physical Activity'] == 'Active' and row['Income Level'] == 'High' else 0,
                axis=1
            )
            print("Created 'activity_income_interaction' feature.")


        self.processed_data_for_ml = df # Update processed data after feature engineering
        return df

    def visualize_data(self, df):
        """
        Generates various plots for data visualization.
        """
        print("\n--- Data Visualization ---")
        # Ensure 'stroke' column exists before plotting
        if 'stroke' not in df.columns:
            print("Warning: 'stroke' column not found for visualization.")
            return

        # Distribution of Age
        plt.figure(figsize=(10, 6))
        sns.histplot(df['age'], kde=True)
        plt.title('Distribution of Age')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.show()

        # Stroke Occurrence Distribution (Pie Chart)
        plt.figure(figsize=(8, 8))
        df['stroke'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral'])
        plt.title('Stroke Occurrence Distribution')
        plt.ylabel('') # Hide the default 'stroke' label
        plt.show()

        # Age Distribution by Stroke Occurrence (Box Plot)
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='stroke', y='age', data=df)
        plt.title('Age Distribution by Stroke Occurrence')
        plt.xlabel('Stroke (0: No, 1: Yes)')
        plt.ylabel('Age')
        plt.show()

        # Stroke Occurrence by Smoking Status (Count Plot)
        if 'smoking_status' in df.columns:
            plt.figure(figsize=(12, 7))
            sns.countplot(x='smoking_status', hue='stroke', data=df, palette='viridis')
            plt.title('Stroke Occurrence by Smoking Status')
            plt.xlabel('Smoking Status')
            plt.ylabel('Count')
            plt.show()

        # Age vs. Average Glucose Level by Stroke Occurrence (Scatter Plot)
        if 'avg_glucose_level' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='age', y='avg_glucose_level', hue='stroke', data=df, palette='coolwarm', alpha=0.7)
            plt.title('Age vs. Average Glucose Level by Stroke Occurrence')
            plt.xlabel('Age')
            plt.ylabel('Average Glucose Level')
            plt.show()

        # Correlation Heatmap for numerical features
        plt.figure(figsize=(12, 10))
        numerical_df = df.select_dtypes(include=np.number)
        sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap of Numerical Features')
        plt.show()


        print("Data visualizations complete.")

    def address_class_imbalance(self, X_train, y_train, strategy='smote'):
        """
        Applies resampling techniques to address class imbalance.
        Currently supports 'smote'.
        Returns resampled X_train and y_train.
        """
        print(f"\n--- Addressing Class Imbalance using {strategy.upper()} ---")
        print(f"Original training class distribution: {Counter(y_train)}")

        if strategy == 'smote':
            # Determine n_neighbors based on the minority class size
            # Check if y_train is Series and has a name before trying value_counts
            if isinstance(y_train, pd.Series):
                minority_class_label = y_train.value_counts().idxmin()
                minority_class_count = y_train.value_counts().min()
            else: # Fallback for numpy array or list
                counts = Counter(y_train)
                if not counts: # Handle empty y_train
                    print("Warning: y_train is empty. Cannot apply SMOTE.")
                    return X_train, y_train
                minority_class_label = min(counts, key=counts.get)
                minority_class_count = counts[minority_class_label]


            # SMOTE requires n_neighbors <= n_samples_fit - 1
            # If minority_class_count is 1, n_neighbors must be 0 (which SMOTE doesn't support well)
            # If minority_class_count is 0 or 1, SMOTE cannot be applied effectively.
            if minority_class_count < 2: # SMOTE needs at least 2 samples for default k_neighbors=1
                print(f"Warning: Minority class '{minority_class_label}' has only {minority_class_count} sample(s). SMOTE cannot be applied effectively.")
                print("Returning original X_train, y_train.")
                return X_train, y_train

            # Set n_neighbors to be at most minority_class_count - 1
            # And also ensure it's at least 1, as SMOTE's default is 5.
            smote_n_neighbors = min(5, minority_class_count - 1)
            if smote_n_neighbors == 0: # If minority_class_count is 1, smote_n_neighbors becomes 0, which is invalid
                 smote_n_neighbors = 1 # A minimum valid value for k_neighbors


            print(f"Adjusting SMOTE's n_neighbors to: {smote_n_neighbors} (based on minority class size: {minority_class_count})")
            sampler = SMOTE(random_state=42, k_neighbors=smote_n_neighbors)
        # elif strategy == 'undersample':
        #     sampler = RandomUnderSampler(random_state=42)
        else:
            print("Warning: Invalid imbalance strategy. Returning original data.")
            return X_train, y_train

        # Ensure X_train is ready for SMOTE (all numeric, no NaN)
        # It's crucial that X_train comes here *after* one-hot encoding and any final cleaning
        if X_train.isnull().any().any():
            print("Warning: X_train contains NaN values. SMOTE cannot handle NaN. Attempting to fill with 0.")
            X_train = X_train.fillna(0) # Or a more sophisticated imputation if NaNs are expected here

        # Ensure X_train has a consistent dtype for SMOTE, e.g., float
        X_train = X_train.astype(float)


        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        print(f"Resampled training class distribution: {Counter(y_resampled)}")
        return X_resampled, y_resampled


    def split_dataset(self, df, target_column, test_size=0.2, random_state=42):
        """
        Prepares data for machine learning by dropping 'id', encoding ALL categorical
        features, and splitting into training and test sets.
        Handles the target column specifically.
        """
        print(f"\n--- Splitting Dataset for target: '{target_column}' ---")

        # Drop 'id' column as it's not a feature
        df_processed = df.drop('id', axis=1, errors='ignore').copy()

        # Identify ALL categorical columns (original and engineered)
        # Exclude numerical columns
        all_cols = set(df_processed.columns)
        numerical_cols = set(df_processed.select_dtypes(include=np.number).columns)
        candidate_categorical_cols = list(all_cols - numerical_cols) # Get non-numeric columns

        # Filter out the target column if it's in the categorical list and handle its encoding separately if needed
        categorical_cols_to_encode = []
        for col in candidate_categorical_cols:
            if col == target_column:
                continue # Handle target later
            # Check if the column is actually categorical (object/category dtype) or has few unique string values
            if df_processed[col].dtype == 'object' or df_processed[col].dtype == 'category' or \
               (df_processed[col].nunique() < 50 and df_processed[col].dtype == 'object'): # Heuristic for potentially categorical numbers
                categorical_cols_to_encode.append(col)

        # Perform one-hot encoding on ALL identified categorical columns
        print(f"One-hot encoding categorical columns: {categorical_cols_to_encode}")
        if categorical_cols_to_encode: # Only call get_dummies if there are columns to encode
            df_processed = pd.get_dummies(df_processed, columns=categorical_cols_to_encode, drop_first=True)

        # Separate features (X) and target (y)
        if target_column not in df_processed.columns:
            print(f"Error: Target column '{target_column}' not found after processing.")
            return None, None, None, None

        X = df_processed.drop(target_column, axis=1)
        y = df_processed[target_column]

        # Convert target 'Yes'/'No' or other strings to numerical if needed
        if y.dtype == 'object' or y.dtype == 'bool' or y.dtype == 'category':
            unique_target_values = y.unique()
            if len(unique_target_values) == 2: # Binary classification
                # Ensure consistent mapping for binary targets, e.g., 'Yes'/'No' or 0/1 for stroke.
                # Assuming the "positive" outcome is 1 and "negative" is 0.
                # This needs to be carefully mapped based on your actual data's string values for each target.
                if target_column == 'stroke':
                    # Assuming '1' for stroke, '0' for no stroke. Adjust if your data uses 'Yes'/'No'.
                    y = y.map({1: 1, 0: 0, 'Yes': 1, 'No': 0}).astype(int) # Add 'Yes'/'No' mapping for robustness
                elif 'Yes' in unique_target_values and 'No' in unique_target_values:
                    y = y.map({'Yes': 1, 'No': 0}).astype(int)
                elif len(unique_target_values) == 2: # Generic for other binary string targets
                    # Simple category codes for generic binary, ensure order is consistent
                    y = y.astype('category').cat.codes
                    print(f"Converted binary target '{target_column}' to numerical using cat.codes.")
                else:
                    print(f"Could not convert binary target '{target_column}' to numerical. Keeping original type.")
            else: # Multi-class classification (e.g., 'Income Level', 'Chronic Stress', 'Physical Activity')
                y = y.astype('category').cat.codes # Converts string categories to numerical codes
                print(f"Converted multi-class target '{target_column}' to numerical using cat.codes.")
        else:
            print(f"Target '{target_column}' is already numerical ({y.dtype}). No conversion needed.")


        # Ensure X contains no non-numeric columns before splitting for ML
        non_numeric_cols_in_X = X.select_dtypes(include=['object', 'category']).columns
        if not non_numeric_cols_in_X.empty:
            print(f"Warning: Non-numeric columns found in X before final split: {list(non_numeric_cols_in_X)}. Attempting to dummy encode.")
            # This indicates an issue if they slipped through, but as a fallback, dummy encode them
            X = pd.get_dummies(X, columns=non_numeric_cols_in_X, drop_first=True)


        # Ensure no NaN values in X before splitting/SMOTE
        if X.isnull().any().any():
            print("Warning: NaN values detected in X before splitting. Filling with 0.")
            X = X.fillna(0) # A simple imputation. Consider more sophisticated methods if appropriate.

        # Convert all features to float to ensure consistency for ML models
        X = X.astype(float)


        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y if target_column == 'stroke' else None # Stratify only for stroke if it's imbalanced and binary
        )
        print("Dataset split into training and test sets.")
        return X_train, X_test, y_train, y_test
