import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_and_preprocess_data(filepath):
    # Load dataset
    df = pd.read_csv("dataset/student/student-mat.csv", sep=';')

    # Feature engineering
    df['G_avg'] = (df['G1'] + df['G2']) / 2
    df['alcohol_use'] = df['Dalc'] + df['Walc']
    df['parent_edu'] = (df['Medu'] + df['Fedu']) / 2

    # Selected features
    features = ['age', 'studytime', 'failures', 'G_avg', 'alcohol_use', 'parent_edu',
                'sex', 'address', 'schoolsup', 'higher', 'internet']
    target = 'G3'

    X = df[features]
    y = df[target]

    # Identify column types
    numerical_cols = ['age', 'studytime', 'failures', 'G_avg', 'alcohol_use', 'parent_edu']
    categorical_cols = ['sex', 'address', 'schoolsup', 'higher', 'internet']

    return X, y, numerical_cols, categorical_cols

def create_pipeline(numerical_cols, categorical_cols, model_type='linear', degree=1, alpha=1.0):
    # Column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first'), categorical_cols)
        ])

    # Choose model
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'ridge':
        model = Ridge(alpha=alpha)
    elif model_type == 'lasso':
        model = Lasso(alpha=alpha)
    else:
        raise ValueError("Invalid model_type. Choose 'linear', 'ridge', or 'lasso'.")

    # Polynomial features if needed
    if degree > 1:
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('regressor', model)
        ])
    else:
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

    return pipeline

def evaluate_model(model, X_test, y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Model Evaluation:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"R2 Score: {r2:.2f}")

    return mae, mse, r2

def visualize_results(y_test, y_pred):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([0, 20], [0, 20], color='red', linestyle='--')
    plt.xlabel("Actual G3 Score")
    plt.ylabel("Predicted G3 Score")
    plt.title("Actual vs Predicted Final Grades")
    plt.grid(True)
    plt.show()

def run_student_performance_pipeline(filepath, model_type='linear', degree=1, alpha=1.0):
    # Load and preprocess data
    X, y, numerical_cols, categorical_cols = load_and_preprocess_data(filepath)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create pipeline
    model = create_pipeline(numerical_cols, categorical_cols, model_type, degree, alpha)

    # Train model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate and visualize
    evaluate_model(model, X_test, y_test, y_pred)
    visualize_results(y_test, y_pred)

if __name__ == "__main__":
    # Example usage
    run_student_performance_pipeline(
        filepath="student-mat.csv",
        model_type='ridge',       # Options: 'linear', 'ridge', 'lasso'
        degree=2,                 # Polynomial degree (1 = no polynomial features)
        alpha=1.0                 # Regularization strength for Ridge/Lasso
    )
