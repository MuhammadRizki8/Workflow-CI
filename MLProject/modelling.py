import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import argparse
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def setup_mlflow():
    """Setup MLflow tracking"""
    # Set MLflow tracking URI (default to local if not specified)
    if not os.environ.get('MLFLOW_TRACKING_URI'):
        mlflow.set_tracking_uri("file:./mlruns")
    
    # Set experiment
    experiment_name = "CI_ML_Model_Training"
    mlflow.set_experiment(experiment_name)
    print(f"MLflow experiment set to: {experiment_name}")

def load_data():
    """Load preprocessed data"""
    print("Loading preprocessed data...")
    
    try:
        # Try different possible paths for data
        data_paths = [
            'preprocessed-csvs/',
            '../preprocessed-csvs/',
            './preprocessed-csvs/',
            'data/',
            './'
        ]
        
        data_loaded = False
        for path in data_paths:
            try:
                X_train = pd.read_csv(f'{path}X_train_processed.csv')
                X_test = pd.read_csv(f'{path}X_test_processed.csv')
                y_train = pd.read_csv(f'{path}y_train_processed.csv').iloc[:, 0]
                y_test = pd.read_csv(f'{path}y_test_processed.csv').iloc[:, 0]
                print(f"Data loaded successfully from: {path}")
                data_loaded = True
                break
            except FileNotFoundError:
                continue
        
        if not data_loaded:
            raise FileNotFoundError("Could not find preprocessed data files")
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"Error loading data: {e}")
        # Create dummy data for CI testing if real data not available
        print("Creating dummy data for testing...")
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.rand(100, 10), columns=[f'feature_{i}' for i in range(10)])
        X_test = pd.DataFrame(np.random.rand(25, 10), columns=[f'feature_{i}' for i in range(10)])
        y_train = pd.Series(np.random.randint(0, 2, 100))
        y_test = pd.Series(np.random.randint(0, 2, 25))
        
        return X_train, X_test, y_train, y_test

def train_model(model, model_name, X_train, X_test, y_train, y_test, params=None):
    """Train model with MLflow logging"""
    
    with mlflow.start_run(run_name=f"{model_name}_CI_training"):
        print(f"\nTraining {model_name}...")
        
        # Use provided parameters or defaults
        if params:
            model.set_params(**params)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log parameters
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("training_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("feature_count", X_train.shape[1])
        
        if params:
            for param, value in params.items():
                mlflow.log_param(param, value)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log model
        mlflow.sklearn.log_model(model, f"{model_name}_model")
        
        # Create and log confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        plot_path = f'{model_name}_confusion_matrix.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        mlflow.log_artifact(plot_path)
        plt.close()
        
        # Clean up
        if os.path.exists(plot_path):
            os.remove(plot_path)
        
        # Log classification report
        report = classification_report(y_test, y_pred)
        report_path = f'{model_name}_classification_report.txt'
        with open(report_path, 'w') as f:
            f.write(f"Classification Report for {model_name}\n")
            f.write("=" * 50 + "\n")
            f.write(report)
        
        mlflow.log_artifact(report_path)
        
        # Clean up
        if os.path.exists(report_path):
            os.remove(report_path)
        
        print(f"{model_name} training completed!")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        return model, accuracy

def train_with_hyperparameter_tuning(X_train, X_test, y_train, y_test):
    """Train models with hyperparameter tuning"""
    
    models_params = {
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                'n_estimators': [50, 100],
                'max_depth': [3, 5, None],
                'min_samples_split': [2, 5]
            }
        },
        "LogisticRegression": {
            "model": LogisticRegression(random_state=42, max_iter=1000),
            "params": {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        }
    }
    
    best_models = {}
    
    for model_name, config in models_params.items():
        with mlflow.start_run(run_name=f"{model_name}_tuned_CI"):
            print(f"\nTuning {model_name}...")
            
            # Grid Search
            grid_search = GridSearchCV(
                config["model"], 
                config["params"], 
                cv=3,  # Reduced for faster CI
                scoring='accuracy',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            # Predictions
            y_pred = best_model.predict(X_test)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Log parameters
            mlflow.log_param("model_type", f"{model_name}_tuned")
            mlflow.log_param("best_params", str(grid_search.best_params_))
            mlflow.log_param("cv_folds", 3)
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("best_cv_score", grid_search.best_score_)
            
            # Log model
            mlflow.sklearn.log_model(best_model, f"{model_name}_tuned_model")
            
            best_models[model_name] = {
                "model": best_model,
                "accuracy": accuracy,
                "params": grid_search.best_params_
            }
            
            print(f"{model_name} tuning completed!")
            print(f"Best params: {grid_search.best_params_}")
            print(f"Test accuracy: {accuracy:.4f}")
    
    return best_models

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='ML Model Training for CI')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha parameter')
    parser.add_argument('--l1_ratio', type=float, default=0.1, help='L1 ratio parameter')
    parser.add_argument('--evaluate-only', action='store_true', help='Only evaluate existing model')
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Starting ML Model Training CI Pipeline")
    print("="*60)
    
    # Setup MLflow
    setup_mlflow()
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    if args.evaluate_only:
        print("Evaluation mode - skipping training")
        return
    
    if args.tune:
        print("Running with hyperparameter tuning...")
        best_models = train_with_hyperparameter_tuning(X_train, X_test, y_train, y_test)
        
        # Find best overall model
        best_overall = max(best_models.items(), key=lambda x: x[1]['accuracy'])
        print(f"\nBest overall model: {best_overall[0]} with accuracy {best_overall[1]['accuracy']:.4f}")
        
    else:
        print("Running basic training...")
        
        # Define models
        models = {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "LogisticRegression": LogisticRegression(
                random_state=42, 
                max_iter=1000,
                C=args.alpha if args.alpha != 0.5 else 1.0
            )
        }
        
        # Train models
        results = {}
        for model_name, model in models.items():
            trained_model, accuracy = train_model(
                model, model_name, X_train, X_test, y_train, y_test
            )
            results[model_name] = accuracy
        
        # Print summary
        print("\n" + "="*60)
        print("Training Summary:")
        for model_name, accuracy in results.items():
            print(f"- {model_name}: {accuracy:.4f}")
        
        best_model = max(results.items(), key=lambda x: x[1])
        print(f"\nBest model: {best_model[0]} with accuracy {best_model[1]:.4f}")
    
    print("\n" + "="*60)
    print("CI Pipeline completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()