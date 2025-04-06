import pandas as pd

# Load the dataset
df = pd.read_csv("dataset/student/student-mat.csv", sep=';')  # Make sure it's in your working directory

# Display top 5 rows
print(df.head())

# Get basic info
print(df.info())

# Summary statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import numpy as np

# Step 1: Select features and target
selected_features = ['sex', 'age', 'address', 'studytime', 'failures', 'schoolsup', 'higher', 'internet', 'G1', 'G2']
X = df[selected_features]
y = df['G3']

# Step 2: Split numerical and categorical columns
numerical_cols = ['age', 'studytime', 'failures', 'G1', 'G2']
categorical_cols = ['sex', 'address', 'schoolsup', 'higher', 'internet']

# Step 3: Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ])

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Preprocessing and data split complete!")

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Create a pipeline with preprocessor and model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Fit the model
model.fit(X_train, y_train)

print("Model training complete!")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Predict on test set
y_pred = model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Get feature names after one-hot encoding
onehot_feature_names = model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_cols)
all_feature_names = numerical_cols + list(onehot_feature_names)

# Get model coefficients
coefficients = model.named_steps['regressor'].coef_

# Pair feature names with coefficients
feature_importance = pd.DataFrame({
    'Feature': all_feature_names,
    'Coefficient': coefficients
})

# Sort by absolute value of coefficients
feature_importance['abs_coeff'] = feature_importance['Coefficient'].abs()
feature_importance = feature_importance.sort_values(by='abs_coeff', ascending=False).drop(columns='abs_coeff')

print(feature_importance)


