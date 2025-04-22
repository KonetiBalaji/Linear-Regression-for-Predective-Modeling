import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils import shuffle

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath, sep=';')
    df['G_avg'] = (df['G1'] + df['G2']) / 2
    df['alcohol_use'] = df['Dalc'] + df['Walc']
    df['parent_edu'] = (df['Medu'] + df['Fedu']) / 2

    features = ['age', 'studytime', 'failures', 'G_avg', 'alcohol_use', 'parent_edu',
                'sex', 'address', 'schoolsup', 'higher', 'internet']
    target = 'G3'

    X = df[features]
    y = df[target]

    numerical_cols = ['age', 'studytime', 'failures', 'G_avg', 'alcohol_use', 'parent_edu']
    categorical_cols = ['sex', 'address', 'schoolsup', 'higher', 'internet']

    return X, y, numerical_cols, categorical_cols

def build_model_pipeline(model_type, degree, alpha, numerical_cols, categorical_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first'), categorical_cols)
        ])

    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'ridge':
        model = Ridge(alpha=alpha)
    elif model_type == 'lasso':
        model = Lasso(alpha=alpha)
    else:
        raise ValueError("Invalid model_type. Choose from: 'linear', 'ridge', 'lasso'.")

    if degree > 1:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('regressor', model)
        ])
    else:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

    return pipeline

def evaluate_model(model, X_test, y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logging.info(f"MAE: {mae:.2f}")
    logging.info(f"MSE: {mse:.2f}")
    logging.info(f"R2 Score: {r2:.2f}")

    return mae, mse, r2

def log_results_to_csv(mae, mse, r2, model_type, degree, alpha, filepath="results_log.csv"):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    row = {
        "timestamp": timestamp,
        "model_type": model_type,
        "degree": degree,
        "alpha": alpha,
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "R2": round(r2, 4)
    }

    df = pd.DataFrame([row])
    header = not os.path.exists(filepath)
    df.to_csv(filepath, mode='a', header=header, index=False)
    logging.info(f"Results saved to {filepath}")

def visualize_results(y_test, y_pred, save=False, filename="plots/predictions.png"):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([0, 20], [0, 20], color='red', linestyle='--')
    plt.xlabel("Actual G3 Score")
    plt.ylabel("Predicted G3 Score")
    plt.title("Actual vs Predicted Final Grades")
    plt.grid(True)

    if save:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        logging.info(f"Prediction plot saved to {filename}")
    else:
        plt.show()

def visualize_feature_importance(model, X_train, degree, save=False, filename="plots/feature_importance.png"):
    if degree > 1:
        logging.warning("Skipping SHAP plot: not supported with PolynomialFeatures (degree > 1)")
        try:
            coefs = model.named_steps['regressor'].coef_
            if 'poly' in model.named_steps:
                feature_names = model.named_steps['poly'].get_feature_names_out(
                    model.named_steps['preprocessor'].get_feature_names_out())
            else:
                feature_names = model.named_steps['preprocessor'].get_feature_names_out()
            importance = pd.Series(coefs, index=feature_names)

            plt.figure(figsize=(10, 6))
            importance.plot(kind='barh')
            plt.title("Feature Importance (Coefficient Magnitude)")
            plt.tight_layout()
            if save:
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                plt.savefig(filename)
                logging.info(f"Feature importance plot saved to {filename}")
            else:
                plt.show()
        except Exception as e:
            logging.warning(f"Fallback feature importance plot failed: {e}")
        return

    try:
        X_transformed = model.named_steps['preprocessor'].transform(X_train)
        explainer = shap.Explainer(model.named_steps['regressor'], X_transformed)
        shap_values = explainer(X_transformed)
        shap.plots.beeswarm(shap_values, show=False)

        if save:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename)
            logging.info(f"SHAP plot saved to {filename}")
        else:
            plt.show()
    except Exception as e:
        logging.warning(f"SHAP visualization failed: {e}")

def grid_search(X_train, y_train, numerical_cols, categorical_cols, model_type):
    param_grid = {'regressor__alpha': [0.01, 0.1, 1, 10, 100]}
    pipeline = build_model_pipeline(model_type, degree=1, alpha=1.0,
                                    numerical_cols=numerical_cols, categorical_cols=categorical_cols)
    search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
    search.fit(X_train, y_train)
    logging.info(f"Best parameters: {search.best_params_}")
    return search.best_estimator_

def parse_args():
    parser = argparse.ArgumentParser(description="Student Performance Prediction")
    parser.add_argument('--model_type', type=str, default='ridge', choices=['linear', 'ridge', 'lasso'])
    parser.add_argument('--degree', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--filepath', type=str, default='dataset/student/student-mat.csv')
    parser.add_argument('--tune', action='store_true', help='Run GridSearchCV for hyperparameter tuning')
    parser.add_argument('--save_plot', action='store_true', help='Save prediction and feature importance plots')
    return parser.parse_args()

def main():
    args = parse_args()
    logging.info("Loading and preprocessing data...")
    X, y, numerical_cols, categorical_cols = load_and_preprocess_data(args.filepath)

    X, y = shuffle(X, y, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if args.tune:
        logging.info("Running GridSearchCV...")
        model = grid_search(X_train, y_train, numerical_cols, categorical_cols, args.model_type)
    else:
        logging.info(f"Training {args.model_type} model with degree={args.degree}, alpha={args.alpha}...")
        model = build_model_pipeline(args.model_type, args.degree, args.alpha, numerical_cols, categorical_cols)
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae, mse, r2 = evaluate_model(model, X_test, y_test, y_pred)
    log_results_to_csv(mae, mse, r2, args.model_type, args.degree, args.alpha)

    visualize_results(y_test, y_pred, save=args.save_plot)
    visualize_feature_importance(model, X_train, args.degree, save=args.save_plot)

if __name__ == "__main__":
    main()