# Medical Insurance Charges Prediction

Machine learning project built to predict medical insurance charges based on demographic and health-related features.

## Dataset

The dataset contains the following features:

- age
- sex
- bmi
- children
- smoker
- region

Target:

- charges

The dataset was downloaded from Kaggle using `kagglehub`.

## Project Workflow

The project follows a complete machine learning workflow:

- Data inspection and exploratory analysis
- Duplicate removal
- Stratified train/test split based on `smoker`
- Numerical preprocessing with `StandardScaler`
- Categorical preprocessing with `OneHotEncoder`
- Linear Regression baseline
- Random Forest model
- Cross-validation
- Hyperparameter tuning with `RandomizedSearchCV`
- Final evaluation on the test set

## Results

| Model | Training RMSE | Cross-Validation RMSE |
|---|---:|---:|
| Linear Regression | ~6210 | ~6253 |
| Random Forest | ~1893 | ~5043 |
| Tuned Random Forest | ~4035 | ~4703 |

Final test RMSE:

```text
~3332
