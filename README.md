# Student Performance Prediction using Linear Regression

This project builds a predictive model to estimate students' final grades in a math course based on their personal, social, and academic characteristics. It leverages linear regression techniques with enhancements like polynomial features and regularization.

## Project Highlights

- Linear, Polynomial, Ridge, and Lasso Regression models
- Feature engineering and data preprocessing
- Evaluation metrics: MAE, MSE, and R² Score
- Visualization of actual vs predicted scores
- Structured, reusable pipeline using scikit-learn
- Function-based script for easy experimentation

## Dataset

This project uses the [Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/Student+Performance) from the UCI Machine Learning Repository.

- File used: `student-mat.csv` (math subject grades)
- Target variable: `G3` (final grade)
- 395 student records with 33 attributes

## Features Used

Numerical:
- age
- studytime
- failures
- G1 and G2 (first and second period grades, combined into G_avg)
- alcohol_use (weekday + weekend consumption)
- parent_edu (average of Medu and Fedu)

Categorical:
- sex
- address
- schoolsup
- higher
- internet

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/student-performance-regression.git
cd student-performance-regression
pip install -r requirements.txt

## File Structure
student-performance-regression/
│
├── student_performance_regression.py                      # Main script
├── dataset\student\student-mat.csv                        # Dataset (math grades)
├── requirements.txt                                       # Required Python packages
└── README.md                                              # Project documentation


## Author

**Balaji Koneti**  
Master’s in Computer Science | Data Engineering | AI Engineering | NLP | ML  
[LinkedIn](https://www.linkedin.com/in/balajikoneti)  
[GitHub](https://github.com/KonetiBalaji)

