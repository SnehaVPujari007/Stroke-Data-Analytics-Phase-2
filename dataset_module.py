# dataset_module.py
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import numpy as np

class StrokeDataset:
    def __init__(self, filepath):
        """
        Initializes the dataset module with the path to the CSV file.
        """
        self.filepath = filepath
        self.data = None

    def load_data(self):
        """
        Loads the dataset from the specified CSV file.
        Returns the loaded pandas DataFrame or None if an error occurs.
        """
        print(f"\n--- Loading data from {self.filepath} ---")
        try:
            self.data = pd.read_csv(self.filepath)
            print("Dataset loaded successfully. Shape:", self.data.shape)
            return self.data
        except FileNotFoundError:
            print(f"Error: The file '{self.filepath}' was not found.")
            return None
        except Exception as e:
            print(f"An error occurred while loading the dataset: {e}")
            return None

    def get_data(self):
        """
        Returns the loaded DataFrame.
        """
        return self.data

# Example of how to use this module (for testing purposes)
if __name__ == "__main__":
    # Create a dummy CSV for testing
    dummy_data = {
        'id': range(10),
        'gender': ['Male', 'Female'] * 5,
        'age': np.random.randint(20, 90, 10),
        'hypertension': np.random.randint(0, 2, 10),
        'heart_disease': np.random.randint(0, 2, 10),
        'ever_married': ['Yes', 'No'] * 5,
        'work_type': ['Private', 'Self-employed'] * 5,
        'Residence_type': ['Urban', 'Rural'] * 5,
        'avg_glucose_level': np.random.uniform(70, 250, 10),
        'bmi': np.random.uniform(15, 40, 10),
        'smoking_status': ['never smoked', 'smokes'] * 5,
        'stroke': [0]*8 + [1]*2,
        'Chronic Stress': ['Low', 'High'] * 5,
        'Physical Activity': ['Active', 'Sedentary'] * 5,
        'Income Level': ['High', 'Low'] * 5
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_df.to_csv('test_data.csv', index=False)
    print("Created dummy_data.csv for dataset module testing.")

    dataset_loader = StrokeDataset('test_data.csv')
    df = dataset_loader.load_data()
    if df is not None:
        print("\nFirst 5 rows of loaded data:")
        print(df.head())
    else:
        print("Data loading failed.")
