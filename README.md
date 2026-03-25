# MLX 2.0 Regression Challenge: Predict the Hype

A machine learning regression project for the MLX 2.0 Challenge that predicts a target variable based on 61 features from structured data.

## Project Overview

This project participates in the MLX 2.0 Regression Challenge where the goal is to build predictive models that forecast a continuous target variable. The notebook implements data exploration, preprocessing, feature engineering, model training, and hyperparameter tuning using multiple regression algorithms.

## Dataset

The dataset consists of:
- **Training set**: `train.csv` - labeled data for model training
- **Test set**: `test.csv` - unlabeled data for predictions
- **Sample submission**: `sample_submission.csv` - template for submission format

### Features
- **Total columns**: 62 (61 features + 1 target variable)
- **Features**: Mix of categorical and numerical features
- **Target**: Continuous variable to predict ("target" column)

## Project Structure

```
mlx-2-0-regression-challenge-predict-the-hype.ipynb  # Main analysis notebook
Submission_1.csv                                      # First submission predictions
Data/
├── train.csv                                         # Training dataset
├── test.csv                                          # Test dataset
└── sample_submission.csv                             # Submission template
```

## Key Workflow Steps

### 1. **Data Loading & Exploration**
   - Load training and test datasets
   - Identify dataset structure and columns
   - Analyze feature types (categorical vs numerical)

### 2. **Data Preprocessing**
   - Handle missing values (NaN records)
   - Encode categorical features (LabelEncoder)
   - Feature engineering with domain-specific mappings

### 3. **Feature Engineering**
   - Temporal feature mappings (season, weekday)
   - Lunar phase encoding
   - Frequency-based categorical encoding

### 4. **Model Training**
   - Multiple algorithm comparison:
     - Linear Regression
     - Decision Tree Regressor
     - Random Forest Regressor
     - Gradient Boosting Regressor
     - XGBoost Regressor

### 5. **Hyperparameter Optimization**
   - RandomizedSearchCV for efficient tuning
   - Cross-validation with StratifiedKFold
   - Custom RMSE scoring metric

### 6. **Predictions & Submission**
   - Generate predictions on test set
   - Format predictions for Kaggle submission
   - Save results to CSV

## Requirements

```
numpy
pandas
scikit-learn
xgboost
```

## Installation

```bash
pip install numpy pandas scikit-learn xgboost
```

## Usage

Open the notebook in Jupyter and run cells sequentially to:
1. Load and explore the data
2. Preprocess and engineer features
3. Train and tune models
4. Generate predictions
5. Create submission file

## Models Used

| Model | Library | Status |
|-------|---------|--------|
| Linear Regression | scikit-learn | Baseline |
| Decision Tree | scikit-learn | Quick evaluation |
| Random Forest | scikit-learn | Tested |
| Gradient Boosting | scikit-learn | Tested |
| XGBoost | xgboost | Primary model |

## Evaluation Metric

- **Metric**: Root Mean Squared Error (RMSE)
- **Scoring**: Custom RMSE scorer with scikit-learn

## Files

- `mlx-2-0-regression-challenge-predict-the-hype.ipynb` - Complete analysis and model pipeline
- `Submission_1.csv` - Initial submission with predictions
- `Data/train.csv` - Training data with labels
- `Data/test.csv` - Test data for predictions
- `Data/sample_submission.csv` - Expected submission format

## Notes

- The notebook handles categorical and numerical features separately
- Missing values are identified and processed accordingly
- Cross-validation is used to ensure robust model evaluation
- Hyperparameter tuning improves model performance

## Future Improvements

- Ensemble methods combining multiple models
- Advanced feature engineering
- Stacking or blending techniques
- Fine-tune hyperparameters further

## Competition Details

- **Challenge**: MLX 2.0 Regression Challenge - Predict the Hype
- **Platform**: Kaggle
- **Task**: Regression (predicting continuous values)

---

*Last Updated: March 2026*
