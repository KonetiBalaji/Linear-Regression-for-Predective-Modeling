import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
from datetime import datetime
from typing import Tuple, Dict, Any, List
import joblib
from pathlib import Path
from scipy import stats
print("scipy.stats import test:", hasattr(stats, 'probplot'))

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.utils import shuffle
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.inspection import permutation_importance

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)

class StudentPerformanceModel:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.best_model = None
        self.feature_importance = None
        self.metrics = {}
        
    def load_and_preprocess_data(self, filepath: str) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
        """Load and preprocess the student performance dataset."""
        logging.info("Loading and preprocessing data...")
        df = pd.read_csv(filepath, sep=';')
        
        # Feature engineering
        df['G_avg'] = (df['G1'] + df['G2']) / 2
        df['alcohol_use'] = df['Dalc'] + df['Walc']
        df['parent_edu'] = (df['Medu'] + df['Fedu']) / 2
        df['study_quality'] = df['studytime'] * (1 - df['failures']/4)  # New feature
        df['attendance_score'] = 1 - (df['absences'] / df['absences'].max())  # New feature
        
        # Handle outliers in numerical columns
        numerical_cols = ['age', 'studytime', 'failures', 'G_avg', 'alcohol_use', 
                         'parent_edu', 'study_quality', 'attendance_score']
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df[col] = df[col].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)
        
        features = numerical_cols + ['sex', 'address', 'schoolsup', 'higher', 'internet']
        target = 'G3'
        
        X = df[features]
        y = df[target]
        
        return X, y, numerical_cols, ['sex', 'address', 'schoolsup', 'higher', 'internet']
    
    def build_model_pipeline(self, model_type: str, degree: int, alpha: float,
                           numerical_cols: List[str], categorical_cols: List[str]) -> Pipeline:
        """Build a model pipeline with preprocessing and feature selection."""
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', RobustScaler(), numerical_cols),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
            ])
        
        if model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'ridge':
            model = Ridge(alpha=alpha)
        elif model_type == 'lasso':
            model = Lasso(alpha=alpha)
        elif model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError("Invalid model_type. Choose from: 'linear', 'ridge', 'lasso', 'random_forest', 'gradient_boosting'.")
        
        if degree > 1:
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
                ('feature_selection', SelectKBest(f_regression, k='all')),
                ('regressor', model)
            ])
        else:
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('feature_selection', SelectKBest(f_regression, k='all')),
                ('regressor', model)
            ])
        
        return pipeline
    
    def evaluate_model(self, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series,
                      y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance with multiple metrics."""
        metrics = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2': r2_score(y_test, y_pred),
            'Explained Variance': explained_variance_score(y_test, y_pred)
        }
        
        for metric, value in metrics.items():
            logging.info(f"{metric}: {value:.4f}")
        
        return metrics
    
    def cross_validate_model(self, model: Pipeline, X: pd.DataFrame, y: pd.Series,
                           n_splits: int = 5) -> Dict[str, float]:
        """Perform k-fold cross-validation."""
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        
        cv_metrics = {
            'CV Mean R2': cv_scores.mean(),
            'CV Std R2': cv_scores.std()
        }
        
        for metric, value in cv_metrics.items():
            logging.info(f"{metric}: {value:.4f}")
        
        return cv_metrics
    
    def grid_search(self, X_train: pd.DataFrame, y_train: pd.Series,
                   numerical_cols: List[str], categorical_cols: List[str],
                   model_type: str) -> Pipeline:
        """Perform grid search for hyperparameter tuning."""
        if model_type in ['ridge', 'lasso']:
            param_grid = {
                'regressor__alpha': [0.01, 0.1, 1, 10, 100],
                'feature_selection__k': [5, 10, 'all']
            }
        elif model_type in ['random_forest', 'gradient_boosting']:
            param_grid = {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__max_depth': [None, 10, 20],
                'feature_selection__k': [5, 10, 'all']
            }
        else:
            param_grid = {
                'feature_selection__k': [5, 10, 'all']
            }
        
        pipeline = self.build_model_pipeline(model_type, degree=1, alpha=1.0,
                                           numerical_cols=numerical_cols,
                                           categorical_cols=categorical_cols)
        
        search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
        search.fit(X_train, y_train)
        
        logging.info(f"Best parameters: {search.best_params_}")
        return search.best_estimator_
    
    def visualize_results(self, y_test: pd.Series, y_pred: np.ndarray,
                         save: bool = False, filename: str = "plots/predictions.png"):
        """Create comprehensive visualization of model results."""
        # Create plots directory if it doesn't exist
        if save:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Actual vs Predicted Plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=y_pred)
        plt.plot([0, 20], [0, 20], color='red', linestyle='--')
        plt.xlabel("Actual G3 Score")
        plt.ylabel("Predicted G3 Score")
        plt.title("Actual vs Predicted Final Grades")
        plt.grid(True)
        
        if save:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()
        
        # Residuals Plot
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_pred, y=residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Residuals Plot")
        plt.grid(True)
        
        if save:
            plt.savefig("plots/residuals.png")
            plt.close()
        else:
            plt.show()
        
        # QQ Plot
        plt.figure(figsize=(10, 6))
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title("Q-Q Plot of Residuals")
        
        if save:
            plt.savefig("plots/qq_plot.png")
            plt.close()
        else:
            plt.show()
    
    def visualize_feature_importance(self, model: Pipeline, X_train: pd.DataFrame,
                                   degree: int, save: bool = False,
                                   filename: str = "plots/feature_importance.png"):
        """Create comprehensive feature importance visualization."""
        if save:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        try:
            # Get feature names
            if 'poly' in model.named_steps:
                feature_names = model.named_steps['poly'].get_feature_names_out(
                    model.named_steps['preprocessor'].get_feature_names_out())
            else:
                feature_names = model.named_steps['preprocessor'].get_feature_names_out()
            
            # Get feature importance based on model type
            if hasattr(model.named_steps['regressor'], 'coef_'):
                importance = pd.Series(model.named_steps['regressor'].coef_,
                                     index=feature_names)
                title = "Feature Importance (Coefficient Magnitude)"
            else:
                importance = pd.Series(
                    model.named_steps['regressor'].feature_importances_,
                    index=feature_names)
                title = "Feature Importance (Random Forest)"
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            importance.sort_values().plot(kind='barh')
            plt.title(title)
            plt.tight_layout()
            
            if save:
                plt.savefig(filename)
                plt.close()
            else:
                plt.show()
            
            # SHAP values if applicable
            if not degree > 1:
                try:
                    X_transformed = model.named_steps['preprocessor'].transform(X_train)
                    explainer = shap.Explainer(model.named_steps['regressor'],
                                             X_transformed)
                    shap_values = explainer(X_transformed)
                    
                    plt.figure(figsize=(12, 8))
                    shap.plots.beeswarm(shap_values, show=False)
                    
                    if save:
                        plt.savefig("plots/shap_values.png")
                        plt.close()
                    else:
                        plt.show()
                except Exception as e:
                    logging.warning(f"SHAP visualization failed: {e}")
        
        except Exception as e:
            logging.warning(f"Feature importance visualization failed: {e}")
    
    def save_model(self, model: Pipeline, filename: str = "models/best_model.joblib"):
        """Save the trained model to disk."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        joblib.dump(model, filename)
        logging.info(f"Model saved to {filename}")
    
    def load_model(self, filename: str = "models/best_model.joblib") -> Pipeline:
        """Load a trained model from disk."""
        return joblib.load(filename)

def main():
    parser = argparse.ArgumentParser(description="Improved Student Performance Prediction")
    parser.add_argument('--model_type', type=str, default='ridge',
                       choices=['linear', 'ridge', 'lasso', 'random_forest', 'gradient_boosting'])
    parser.add_argument('--degree', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--filepath', type=str, default='dataset/student/student-mat.csv')
    parser.add_argument('--tune', action='store_true',
                       help='Run GridSearchCV for hyperparameter tuning')
    parser.add_argument('--save_plot', action='store_true',
                       help='Save prediction and feature importance plots')
    parser.add_argument('--cross_validate', action='store_true',
                       help='Perform k-fold cross-validation')
    args = parser.parse_args()
    
    # Initialize model
    model = StudentPerformanceModel(vars(args))
    
    # Load and preprocess data
    X, y, numerical_cols, categorical_cols = model.load_and_preprocess_data(args.filepath)
    
    # Split data
    X, y = shuffle(X, y, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                       random_state=42)
    
    # Train model
    if args.tune:
        logging.info("Running GridSearchCV...")
        best_model = model.grid_search(X_train, y_train, numerical_cols,
                                     categorical_cols, args.model_type)
    else:
        logging.info(f"Training {args.model_type} model with degree={args.degree}, "
                    f"alpha={args.alpha}...")
        best_model = model.build_model_pipeline(args.model_type, args.degree,
                                              args.alpha, numerical_cols,
                                              categorical_cols)
        best_model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = best_model.predict(X_test)
    metrics = model.evaluate_model(best_model, X_test, y_test, y_pred)
    
    # Cross-validation if requested
    if args.cross_validate:
        cv_metrics = model.cross_validate_model(best_model, X, y)
    
    # Visualize results
    model.visualize_results(y_test, y_pred, save=args.save_plot)
    model.visualize_feature_importance(best_model, X_train, args.degree,
                                     save=args.save_plot)
    
    # Save model
    model.save_model(best_model)

if __name__ == "__main__":
    main() 