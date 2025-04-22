# Student Performance Prediction using Linear Regression

This project builds a predictive model to estimate students' final math grades using various linear regression techniques, enhanced with polynomial features, regularization, hyperparameter tuning, cross-validation, and SHAP-based model interpretability.

## Features

- Linear, Polynomial, Ridge, and Lasso Regression
- Hyperparameter tuning with GridSearchCV
- Cross-validation (5-fold)
- SHAP visualizations for feature importance
- Command-line arguments for flexible experimentation
- Logging and modular design

## Dataset

- Source: [UCI Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/Student+Performance)
- File: `student-mat.csv` (math grades only)

## Installation

```bash
git clone https://github.com/KonetiBalaji/Linear-Regression-for-Predective-Modeling.git
cd student-performance-regression
pip install -r requirements.txt
```

## Usage

```bash
# Basic training
python student_performance_regression.py --model_type ridge --degree 2 --alpha 1.0

# Run hyperparameter tuning
python student_performance_regression.py --model_type ridge --tune
```

## File Structure

```
student-performance-regression/
├── dataset/
│   └── student/
│       └── student-mat.csv
├── student_performance_regression.py
├── requirements.txt
├── README.md
└── LICENSE
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Author

**Balaji Koneti**  
Master’s in Computer Science | Data Engineering | AI Engineering  
[LinkedIn](https://www.linkedin.com/in/balajikoneti)  
[GitHub](https://github.com/KonetiBalaji)