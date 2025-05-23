# ml_model_module.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler # For SVC or K-Neighbors
import warnings
warnings.filterwarnings("ignore")


class MLModel:
    def __init__(self, X_train, X_test, y_train, y_test):
        """
        Initializes the MLModel with training and testing data.
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models = {}
        self.results = {} # Stores metrics for each model and target
        self.scalers = {} # To store scalers if needed for specific models

    def _preprocess_data_for_model(self, model_name, X_train_input, X_test_input):
        """
        Applies specific preprocessing (e.g., scaling) if required by the model.
        Returns processed X_train and X_test.
        """
        if model_name in ['SVC', 'KNeighborsClassifier']:
            # Use a new scaler instance or retrieve a stored one if already fitted
            if model_name not in self.scalers:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_input)
                self.scalers[model_name] = scaler # Store the fitted scaler
            else:
                scaler = self.scalers[model_name]
                X_train_scaled = scaler.transform(X_train_input) # Just transform if scaler already fitted

            X_test_scaled = scaler.transform(X_test_input)
            print(f"Scaling data for {model_name}...")
            return X_train_scaled, X_test_scaled
        return X_train_input, X_test_input


    def train_and_evaluate(self, target_name):
        """
        Trains and evaluates multiple classification models for a given target.
        Stores results in self.results.
        """
        print(f"\n--- Training and Evaluating Models for Target: '{target_name}' ---")

        # Define the models to be used
        # To meet the "3 models" criteria, uncomment two more models from the list below.
        model_definitions = {
            'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear', max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100 , n_jobs = 1),
            # 'SVC': SVC(random_state=42, probability=True), # Add SVC, requires scaling
            # 'Gradient Boosting': GradientBoostingClassifier(random_state=42) # Add Gradient Boosting
            # 'K-Neighbors': KNeighborsClassifier(), # Add K-Neighbors, requires scaling
            # 'Gaussian Naive Bayes': GaussianNB() # Add Naive Bayes
        }

        self.results[target_name] = {} # Initialize results for the current target

        for model_name, model in model_definitions.items():
            print(f"\n--- Training {model_name} ---")

            # Apply model-specific preprocessing (X_train and X_test used for THIS model's training)
            X_train_processed, X_test_processed_for_model = self._preprocess_data_for_model(model_name, self.X_train, self.X_test)

            try:
                model.fit(X_train_processed, self.y_train)
                y_pred = model.predict(X_test_processed_for_model)

                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
                cm = confusion_matrix(self.y_test, y_pred)
                class_report = classification_report(self.y_test, y_pred, zero_division=0)

                print(f"Accuracy: {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"F1-Score: {f1:.4f}")
                print("\nConfusion Matrix:\n", cm)
                print("\nClassification Report:\n", class_report)

                self.models[f"{target_name}_{model_name}"] = model
                
                # Store all relevant data, including y_prob and X_test_processed_for_model
                model_metrics = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'confusion_matrix': cm,
                    'classification_report': class_report,
                    'y_pred': y_pred, # Store predictions
                    'X_test_processed': X_test_processed_for_model # Store the processed X_test used for this model
                }

                # Store ROC AUC if it's a binary classification and model supports probability prediction
                # Ensure y_test is binary and has more than one class to avoid errors in roc_curve
                if len(np.unique(self.y_test)) == 2 and len(np.unique(self.y_test)) > 1 and hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test_processed_for_model)[:, 1]
                    fpr, tpr, _ = roc_curve(self.y_test, y_prob)
                    roc_auc = auc(fpr, tpr)
                    model_metrics['roc_auc'] = roc_auc
                    model_metrics['y_prob'] = y_prob # Store probabilities
                    print(f"ROC AUC: {roc_auc:.4f}")
                else:
                    print(f"ROC AUC not applicable or model does not support predict_proba for {target_name} or y_test is not binary.")

                self.results[target_name][model_name] = model_metrics

            except Exception as e:
                print(f"Error training/evaluating {model_name}: {e}")
                self.results[target_name][model_name] = {'error': str(e)}

    def visualize_model_performance(self):
        """
        Visualizes the performance metrics across different models and targets.
        This method will iterate through self.results to generate plots.
        """
        if not self.results:
            print("No model results to visualize. Please train models first.")
            return

        print("\n--- Visualizing Model Performance ---")

        for target_name, models_results in self.results.items():
            if not models_results:
                print(f"No results for target: {target_name}")
                continue

            # Prepare data for plotting
            model_names = []
            accuracies = []
            precisions = []
            recalls = []
            f1_scores = []
            roc_aucs = [] # To store ROC AUC if available

            for model_name, metrics in models_results.items():
                if 'error' in metrics:
                    print(f"Skipping visualization for {model_name} due to error: {metrics['error']}")
                    continue
                model_names.append(model_name)
                accuracies.append(metrics.get('accuracy', 0))
                precisions.append(metrics.get('precision', 0))
                recalls.append(metrics.get('recall', 0))
                f1_scores.append(metrics.get('f1_score', 0))
                roc_aucs.append(metrics.get('roc_auc', np.nan)) # Append NaN if ROC AUC not applicable

            if not model_names:
                print(f"No valid models to visualize for target: {target_name}")
                continue

            # Filter metrics for plotting (only common ones for bar chart)
            metrics_to_plot_df = pd.DataFrame({
                'Model': model_names,
                'Accuracy': accuracies,
                'Precision': precisions,
                'Recall': recalls,
                'F1-Score': f1_scores
            }).set_index('Model')

            print(f"\nPerformance Summary for '{target_name}':")
            print(metrics_to_plot_df)


            # Bar plot for performance metrics
            fig, ax = plt.subplots(figsize=(12, 7))
            metrics_to_plot_df.plot(kind='bar', ax=ax, colormap='viridis')
            plt.title(f'Model Performance Comparison for {target_name}')
            plt.ylabel('Score')
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 1.05)
            plt.legend(title='Metric')
            plt.tight_layout()
            plt.show()

            # ROC Curve for binary classification targets (only if 'stroke' and ROC AUCs were calculated)
            # Check if there are any valid ROC AUCs and target is 'stroke'
            if target_name == 'stroke' and not np.all(np.isnan(roc_aucs)):
                plt.figure(figsize=(8, 7))
                for model_name, metrics in models_results.items():
                    if 'y_prob' in metrics and 'roc_auc' in metrics: # Ensure y_prob and roc_auc are stored
                        # Use the stored y_prob directly
                        fpr, tpr, _ = roc_curve(self.y_test, metrics['y_prob'])
                        roc_auc = metrics['roc_auc']
                        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curves for {target_name} Prediction')
                plt.legend()
                plt.grid(True)
                plt.show()
            elif target_name == 'stroke' and np.all(np.isnan(roc_aucs)):
                print(f"ROC Curves for '{target_name}' could not be plotted as no valid ROC AUCs were found.")
            else:
                print(f"ROC Curves are typically plotted for binary classification targets like 'stroke'. Skipping for '{target_name}'.")

        print("Model performance visualizations complete.")

# Example of how to use this module (for testing purposes)
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split # Make sure this is imported

    # Generate some synthetic data for testing
    # Binary classification with imbalance
    X, y = make_classification(n_samples=100, n_features=10, n_informative=5,
                               n_redundant=0, n_clusters_per_class=1,
                               weights=[0.8, 0.2], flip_y=0, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    y_series = pd.Series(y, name='stroke')

    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_df, y_series, test_size=0.3, random_state=42, stratify=y_series)

    ml_model_test = MLModel(X_train_t, X_test_t, y_train_t, y_test_t)
    ml_model_test.train_and_evaluate(target_name='stroke')
    ml_model_test.visualize_model_performance()

    # Test with multi-class target
    X_multi, y_multi = make_classification(n_samples=100, n_features=10, n_informative=5,
                                           n_redundant=0, n_clusters_per_class=1,
                                           n_classes=3, random_state=42)
    X_multi_df = pd.DataFrame(X_multi, columns=[f'feature_{i}' for i in range(10)])
    y_multi_series = pd.Series(y_multi, name='Income Level')
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi_df, y_multi_series, test_size=0.3, random_state=42)

    ml_model_test_multi = MLModel(X_train_m, X_test_m, y_train_m, y_test_m)
    ml_model_test_multi.train_and_evaluate(target_name='Income Level')
    ml_model_test_multi.visualize_model_performance()
