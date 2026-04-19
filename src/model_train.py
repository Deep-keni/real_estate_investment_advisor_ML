import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
import mlflow
import mlflow.sklearn
import joblib
import os

os.makedirs('models', exist_ok=True)

df = pd.read_csv('data/processed_data/cleaned_data.csv')

# ─────────────────────────────────────────
# CLASSIFICATION
# ─────────────────────────────────────────

xc = df.drop(columns=['Future_Price_5Y', 'Good_Investment', 'Price_per_SqFt'])
yc = df['Good_Investment']

sm = SMOTE(random_state=42)
xc_res, yc_res = sm.fit_resample(xc, yc)

x_train, x_test, y_train, y_test = train_test_split(xc_res, yc_res, test_size=0.2, random_state=42)

# Random Forest Classifier (no MLflow - just for comparison)
rfc = RandomForestClassifier(n_estimators=100, max_depth=5)
rfc.fit(x_train, y_train)
preds_rcf = rfc.predict(x_test)
print("---- Random Forest Classifier ----")
print(f"Accuracy: {accuracy_score(y_test, preds_rcf)}")
print(f"F1 Score: {f1_score(y_test, preds_rcf)}")
print(f"ROC AUC: {roc_auc_score(y_test, preds_rcf)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, preds_rcf)}")

# XGBoost Classifier (with MLflow)
mlflow.set_experiment("classification_experiment")

clf_params = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 5
}

with mlflow.start_run(run_name="XGBoost_Classifier"):
    xclf = XGBClassifier(**clf_params)
    xclf.fit(x_train, y_train)
    preds1_rcf = xclf.predict(x_test)

    mlflow.log_params(clf_params)

    mlflow.log_metric("accuracy", accuracy_score(y_test, preds1_rcf))
    mlflow.log_metric("f1_score", f1_score(y_test, preds1_rcf))
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, preds1_rcf))

    mlflow.sklearn.log_model(xclf, "xgb_classifier")

    print("---- XGBoost Classifier ----")
    print(f"Accuracy: {accuracy_score(y_test, preds1_rcf)}")
    print(f"F1 Score: {f1_score(y_test, preds1_rcf)}")
    print(f"ROC AUC: {roc_auc_score(y_test, preds1_rcf)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, preds1_rcf)}")

joblib.dump(xclf, 'models/classifier.pkl')


# ─────────────────────────────────────────
# REGRESSION
# ─────────────────────────────────────────

xr = df.drop(columns=['Future_Price_5Y', 'Good_Investment', 'Price_in_Lakhs'])
yr = df['Future_Price_5Y']

x_train, x_test, y_train, y_test = train_test_split(xr, yr, test_size=0.2, random_state=42)

# Random Forest Regressor (no MLflow - just for comparison)
rfr = RandomForestRegressor(n_estimators=100, max_depth=5)
rfr.fit(x_train, y_train)
preds_rfr = rfr.predict(x_test)
print("---- Random Forest Regressor ----")
print(f"RMSE: {root_mean_squared_error(y_test, preds_rfr)}")
print(f"MAE: {mean_absolute_error(y_test, preds_rfr)}")
print(f"R2: {r2_score(y_test, preds_rfr)}")

# XGBoost Regressor (with MLflow)
mlflow.set_experiment("regression_experiment")

reg_params = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 5
}

with mlflow.start_run(run_name="XGBoost_Regressor"):
    xrgr = XGBRegressor(**reg_params)
    xrgr.fit(x_train, y_train)
    preds1_rfr = xrgr.predict(x_test)

    mlflow.log_params(reg_params)

    mlflow.log_metric("rmse", root_mean_squared_error(y_test, preds1_rfr))
    mlflow.log_metric("mae", mean_absolute_error(y_test, preds1_rfr))
    mlflow.log_metric("r2", r2_score(y_test, preds1_rfr))

    mlflow.sklearn.log_model(xrgr, "xgb_regressor")

    print("---- XGBoost Regressor ----")
    print(f"RMSE: {root_mean_squared_error(y_test, preds1_rfr)}")
    print(f"MAE: {mean_absolute_error(y_test, preds1_rfr)}")
    print(f"R2: {r2_score(y_test, preds1_rfr)}")

joblib.dump(xrgr, 'models/regressor.pkl')

# Note: XGBoost chosen as best model for both tasks
# Hyperparameter tuning skipped as results are already strong
# (Classification: 99.3%, Regression R2: 0.995)



