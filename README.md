## Introduction

The MLX 2.0 Regression competition focuses on predicting the popularity of songs using real-world music data. The objective is to develop a regression model that can accurately predict a song popularity score ranging from 0 to 100. This score reflects listener reception and overall performance on platforms such as Billboard charts.

The dataset includes a wide range of features, such as audio characteristics (energy, danceability, tonal patterns), artist statistics, and track metadata. Since song popularity is influenced by multiple complex and often non-linear factors, machine learning regression methods are applied to model these relationships.

In this project, different regression approaches were explored, feature engineering and preprocessing were applied, and models were evaluated using standard metrics. The goal was to build a robust model that generalizes well to unseen data and performs strongly on the Kaggle leaderboard.

## Dataset Description

For this project, two datasets were provided:

- **Training dataset**: contains the target variable (song popularity score)
- **Test dataset**: does not contain the target variable and is used for final predictions

Both datasets include **61 input features**:

- **52 numerical features**
- **9 categorical features**

Examples of numerical features:

- `id`
- `emotional_charge_2`
- `groove_efficiency_1`
- `beat_frequency_1`
- `organic_texture_2`
- `harmonic_scale_1`
- `intensity_index_0`

Examples of categorical features:

- `composition_label_0`
- `composition_label_1`
- `publication_timestamp`
- `weekday_of_release`
- `season_of_release`
- `lunar_phase`
- `creator_collective`

### Missing Value Statistics

- **Training data**: 61,609 total records, 59,828 records with at least one missing value
- **Test data**: 41,074 total records, 39,864 records with at least one missing value
- **Missing-value spread**: 60 out of 61 features contain missing values

Handling missing data was a critical preprocessing requirement.

## Data Preprocessing

1. Duplicate rows were removed.
2. Missing values were imputed instead of dropping rows to preserve data volume and patterns.

### Numerical Feature Imputation

- Missing numerical values were replaced using the **mean**.
- Means were calculated after concatenating numerical columns from both train and test datasets.

### Categorical Feature Imputation

- `weekday_of_release` and `season_of_release` were partially derived from `publication_timestamp` when possible.
- Remaining missing categorical values were filled using the **mode** (most frequent value), calculated on concatenated train+test categorical data.

### Categorical Encoding

- **Label Encoding** for low-cardinality columns:
  - `weekday_of_release`
  - `season_of_release`
  - `lunar_phase`
- **Frequency Encoding** for high-cardinality columns:
  - `composition_label_0`
  - `composition_label_1`
  - `composition_label_2`
  - `creator_collective`
  - `track_identifier`

## Feature Engineering

Several new features were created to improve predictive performance:

1. **Timestamp decomposition**:
   - `publication_timestamp` was split into year, month, and day.
2. **Tonal variation feature**:
   - Standard deviation of `tonal_mode_0`, `tonal_mode_1`, and `tonal_mode_2`.
   - $$\sigma = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2}$$
3. **Intensity consistency**:
   - Mean of `intensity_index_0`, `intensity_index_1`, and `intensity_index_2`.
   - $$IC = \frac{I_0 + I_1 + I_2}{3}$$
4. **Overall energy features**:
   - `overall_energy_0`, `overall_energy_1`, `overall_energy_2` created as:
   - $$OE_i = \frac{I_i + G_i + R_i}{3}, \quad i \in \{0,1,2\}$$

## Feature Selection

Using correlation analysis, the following features were removed due to lower usefulness or redundancy:

- `groove_efficiency_1`
- `beat_frequency_1`
- `intensity_index_0`
- `groove_efficiency_2`
- `composition_label_0_encoded`
- `time_signature_2`
- `tonal_mode_1`
- `publication_month`
- `harmonic_scale_0`
- `tonal_mode_0`
- `duration_ms_1`
- `organic_texture_1`

This reduced noise and improved model performance.

## Evaluation Metrics

The following metrics were used on the validation set:

1. **Mean Absolute Error (MAE)**
   - $$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$
2. **Root Mean Squared Error (RMSE)**
   - $$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$
3. **R-squared Score (R2)**
   - $$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

The competition ranking metric is **RMSE**.

## Model Selection

The training dataset was split into:

- **90% training**
- **10% validation**

Tested models:

- Linear Regression
- Decision Tree Regression
- Random Forest Regression
- Gradient Boosting Regression
- XGBoost Regression
- K-Nearest Neighbors (KNN) Regression

### Validation Results

| Model | MAE | RMSE | R-squared |
|---|---:|---:|---:|
| Linear Regression | 15.89008 | 19.57039 | 0.17259 |
| Decision Tree | 7.38094 | 11.77477 | 0.70049 |
| Random Forest | 5.32355 | 9.25780 | 0.81484 |
| Gradient Boosting | 8.88880 | 11.61224 | 0.70869 |
| XGBoost | 0.70869 | 9.77395 | 0.79362 |
| K-Nearest Neighbors | 8.89326 | 15.61243 | 0.47342 |

Based on these results, **Random Forest Regression** was selected as the best-performing model.

## Hyperparameter Tuning

Best Random Forest hyperparameters:

- `n_estimators = 1000`
- `max_depth = 100`
- `min_samples_split = 2`
- `min_samples_leaf = 1`
- `random_state = 42`

These values gave a good bias-variance balance and improved generalization.

## Conclusion

This project successfully built a robust regression pipeline for predicting song popularity from structured music data. Strong performance was achieved through:

- careful missing-value handling
- mixed encoding strategies for categorical features
- targeted feature engineering
- correlation-based feature selection
- model comparison and tuning

Among all tested models, Random Forest performed best on validation metrics and delivered competitive leaderboard scores.

## Kaggle Results

- **Public score:** 9.1032
- **Private score:** 9.0739

## End-to-End Workflow Summary

1. Data loading and schema inspection
2. Missing value analysis and imputation
3. Categorical encoding (label + frequency)
4. Feature engineering and feature filtering
5. Multi-model training and validation comparison
6. Best-model tuning and Kaggle submission generation

## Tools and Environment

- **Language:** Python
- **Primary libraries:** pandas, numpy, scikit-learn, xgboost
- **Notebook:** `mlx-2-0-regression-challenge-predict-the-hype.ipynb`
- **Evaluation platform:** Kaggle

## Reproducibility Notes

- A fixed random seed (`random_state = 42`) was used in model training.
- Train/validation split was kept constant at 90:10 for fair model comparison.
- The same preprocessing pipeline was applied to both train and test datasets.
- Submission formatting followed `Data/sample_submission.csv`.

## Key Challenges Encountered

- Extremely high missing-value density across features
- Mixed low-cardinality and high-cardinality categorical variables
- Balancing model complexity and generalization performance
- Avoiding overfitting during hyperparameter tuning

## Limitations

- Mean and mode imputation may not fully preserve complex feature distributions.
- A single train/validation split can introduce variance in metric estimates.
- Frequency encoding can lose semantic relationships between categories.

## Future Improvements

- K-fold cross-validation for more stable performance estimation
- Advanced imputers (for example, KNN or iterative imputation)
- Target encoding with leakage-safe validation strategy
- Ensemble or stacking methods to improve leaderboard score
- SHAP-based interpretability for stronger feature-level insights

## Project Files

- `mlx-2-0-regression-challenge-predict-the-hype.ipynb`
- `README.md`
- `submission.csv`
- `Data/train.csv`
- `Data/test.csv`
- `Data/sample_submission.csv`
