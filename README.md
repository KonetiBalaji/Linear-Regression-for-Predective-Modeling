# Improved Student Performance Prediction

This project implements an enhanced version of the student performance prediction model with several improvements over the original implementation.

## Key Improvements

1. **Enhanced Feature Engineering**
   - Added new features: study_quality and attendance_score
   - Implemented outlier handling using IQR method
   - Added feature selection using SelectKBest

2. **Advanced Models**
   - Added Random Forest and Gradient Boosting models
   - Implemented proper model comparison
   - Added support for polynomial features

3. **Improved Evaluation**
   - Added more evaluation metrics (RMSE, Explained Variance)
   - Implemented k-fold cross-validation
   - Added comprehensive model validation plots

4. **Better Visualization**
   - Added residuals plot
   - Added Q-Q plot for normality check
   - Enhanced feature importance visualization
   - Added SHAP values for model interpretability

5. **Code Quality**
   - Implemented proper class structure
   - Added type hints
   - Improved error handling
   - Added comprehensive logging
   - Added model persistence

## Project Files: Original vs Improved

- **student_performance_regression.py**
  - The original version of the project.
  - Implements basic linear, ridge, and lasso regression models.
  - Includes basic preprocessing, feature engineering, and evaluation.
  - Supports polynomial features and SHAP-based interpretability.
  - Suitable for learning the fundamentals of regression modeling and pipeline construction.

- **student_performance_regression_improved.py**
  - The improved and advanced version of the project.
  - Adds advanced models (Random Forest, Gradient Boosting), robust feature engineering, and outlier handling.
  - Includes feature selection, k-fold cross-validation, and more evaluation metrics.
  - Provides enhanced visualizations (residuals, Q-Q plot, feature importance).
  - Uses a class-based structure, better logging, and model persistence.
  - Recommended for portfolio use and demonstrating best practices in applied machine learning.

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage:
```bash
python student_performance_regression_improved.py
```

Advanced usage with options:
```bash
python student_performance_regression_improved.py \
    --model_type random_forest \
    --tune \
    --cross_validate \
    --save_plot
```

### Command Line Arguments

- `--model_type`: Type of model to use (linear, ridge, lasso, random_forest, gradient_boosting)
- `--degree`: Degree of polynomial features (default: 1)
- `--alpha`: Regularization parameter for ridge/lasso (default: 1.0)
- `--filepath`: Path to the dataset (default: dataset/student/student-mat.csv)
- `--tune`: Run GridSearchCV for hyperparameter tuning
- `--save_plot`: Save prediction and feature importance plots
- `--cross_validate`: Perform k-fold cross-validation

## Output

The script generates:
1. Model performance metrics in the console
2. Visualization plots in the `plots/` directory (if --save_plot is used)
3. Trained model saved in the `models/` directory
4. Training logs in `model_training.log`

## Model Performance

The improved version includes:
- Better feature engineering for improved predictions
- More robust model evaluation through cross-validation
- Enhanced model interpretability through SHAP values
- Comprehensive visualization of model performance

## Contributing

Feel free to submit issues and enhancement requests!

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
Master's in Computer Science | Data Engineering | AI Engineering  
[LinkedIn](https://www.linkedin.com/in/balaji-koneti)  
[GitHub](https://github.com/KonetiBalaji)