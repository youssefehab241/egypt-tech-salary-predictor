# ============================================================
#   Salary Prediction — ML Training Script
#   Input: salaries_cleaned.csv
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("--- Loading Data ---")
df = pd.read_csv('salaries_cleaned.csv')
df_clean = df.copy()
print(f"Loaded: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")
print(f"Salary range: {df_clean['salary_mid'].min():,.0f} – {df_clean['salary_mid'].max():,.0f} EGP")


# ============================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================
print("\n--- Feature Engineering ---")

if "seniority" not in df_clean.columns:
    def get_seniority(years):
        if years <= 1.5: return "junior"
        elif years <= 4: return "mid_senior"
        elif years <= 8: return "senior"
        else:            return "lead"
    df_clean["seniority"] = df_clean["experience_years"].apply(get_seniority)

EGYPT_CITIES   = {"Cairo","Alexandria","Giza","Mansoura","Tanta","Assiut","Zagazig","Suez","Ismailia","Luxor","Aswan","Port Said"}
FOREIGN_CITIES = {"Dubai","Abu Dhabi","Sharjah","Riyadh","Jeddah","Saudi Arabia","Kuwait","Qatar","Germany","UK","USA"}

def simplify_city(city):
    if pd.isna(city) or city == "Unknown": return "Unknown"
    if city == "Cairo":                    return "Cairo"
    if city == "Alexandria":               return "Alexandria"
    if city == "Giza":                     return "Giza"
    if city in FOREIGN_CITIES:             return "Foreign"
    return "Other Egypt"

df_clean["city_group"] = df_clean["city"].apply(simplify_city)


# ============================================================
# STEP 3: ONE-HOT ENCODING
# ============================================================
print("\n--- Encoding ---")

df_encoded = pd.get_dummies(
    df_clean.drop(columns=["city"]),
    columns=["job_title", "source_name", "seniority", "work_mode", "city_group"]
)

bool_cols = df_encoded.select_dtypes(include="bool").columns
df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

# Interaction features: experience × job title
job_cols = [c for c in df_encoded.columns if "job_title_" in c]
for col in job_cols:
    name = f"exp_x_{col.replace('job_title_', '').replace(' ', '_')}"
    df_encoded[name] = df_encoded["experience_years"] * df_encoded[col]

print(f"Final shape: {df_encoded.shape}")


# ============================================================
# STEP 4: TRAIN / TEST SPLIT
# ============================================================
X = df_encoded.drop(columns=["salary_mid"])
y = df_encoded["salary_mid"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42
)
print(f"Train: {len(X_tr):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")


# ============================================================
# STEP 5: BASELINE — LINEAR REGRESSION
# ============================================================
print("\n--- Baseline: Linear Regression ---")

lr_baseline = LinearRegression()
lr_baseline.fit(X_train, np.log1p(y_train))
print("Linear Regression done.")


# ============================================================
# STEP 5.5: RIDGE REGRESSION
# ============================================================
#
# from sklearn.metrics import mean_absolute_error
# from sklearn.linear_model import Ridge
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
#
# best_alpha = None
# best_score = float("inf")
# best_ridge_model = None
#
# print("\n--- Ridge Alpha Tuning ---")
# alphas = [0.01, 0.1, 1, 10, 50, 100]
# for a in alphas:
#
#     ridge_model = Pipeline([
#         ("scaler", StandardScaler()),
#         ("ridge", Ridge(alpha=a))
#     ])
#
#     ridge_model.fit(X_tr, np.log1p(y_tr))
#
#     val_pred = ridge_model.predict(X_val)
#     val_pred = np.expm1(val_pred)
#
#     score = mean_absolute_error(y_val, val_pred)
#
#     print(f"alpha={a:<5}  Val MAE={score:,.2f}")
#
#     if score < best_score:
#         best_score = score
#         best_alpha = a
#         best_ridge_model = ridge_model
#
# print("\nBest alpha:", best_alpha)


from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

print("\n--- Ridge Regression (GridSearchCV) ---")

# log transform target (same as other models)
y_tr_log = np.log1p(y_tr)
y_val_log = np.log1p(y_val)
y_test_log = np.log1p(y_test)

# pipeline: scaling + ridge
ridge_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge())
])

# hyperparameter grid
param_grid = {
    "ridge__alpha": [0.01, 0.1, 1, 10, 50, 100, 200]
}

# Grid Search CV
ridge_grid = GridSearchCV(
    ridge_pipe,
    param_grid,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=1
)

# train ONLY on training split
ridge_grid.fit(X_tr, y_tr_log)

best_ridge = ridge_grid.best_estimator_
print("\nBest Ridge alpha:", ridge_grid.best_params_)




# ============================================================
# STEP 6: XGBOOST WITH RANDOMIZED SEARCH
# ============================================================
print("\n--- Tuning XGBoost (RandomizedSearch) ---")

param_dist = {
    "n_estimators"    : [300, 500],
    "max_depth"       : [3, 4, 5, 6],
    "learning_rate"   : [0.01, 0.05, 0.1],
    "reg_alpha"       : [0, 0.1, 1],
    "subsample"       : [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
}

xgb_model = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_estimators=500,
    early_stopping_rounds=20,
    eval_metric="mae",
)

random_search = RandomizedSearchCV(
    xgb_model, param_dist,
    n_iter=10, cv=3,
    scoring="neg_mean_absolute_error",
    n_jobs=-1, verbose=1, random_state=42
)
random_search.fit(
    X_tr, np.log1p(y_tr),
    eval_set=[(X_val, np.log1p(y_val))],
    verbose=False
)
best_model = random_search.best_estimator_
print(f"\nBest params: {random_search.best_params_}")

joblib.dump(best_model, "final_model.pkl")

# ============================================================
# STEP 7: EVALUATION
# ============================================================
print("\n--- Evaluating Models ---")

lr_pred  = np.expm1(lr_baseline.predict(X_test))
xgb_pred = np.expm1(best_model.predict(X_test))

ridge_pred_log = best_ridge.predict(X_test)
ridge_pred = np.expm1(ridge_pred_log)



def get_metrics(y_true, y_pred):
    return {
        "MAE" : mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2"  : r2_score(np.log1p(y_true), np.log1p(np.maximum(y_pred, 1)))
    }

m_lr  = get_metrics(y_test, lr_pred)
m_ridge = get_metrics(y_test, ridge_pred)
m_xgb = get_metrics(y_test, xgb_pred)

print("\n" + "=" * 55)
print("FINAL RESULTS")
print("=" * 55)
print(f"{'Model':<28} {'MAE':>8} {'RMSE':>10} {'R²':>8}")
print("-" * 55)
print(f"{'Linear Regression':<28} {m_lr['MAE']:>8,.0f} {m_lr['RMSE']:>10,.0f} {m_lr['R2']:>8.4f}")
print(f"{'XGBoost (Tuned)':<28} {m_xgb['MAE']:>8,.0f} {m_xgb['RMSE']:>10,.0f} {m_xgb['R2']:>8.4f}")
print("=" * 55)

#

import pandas as pd
# # ============================================================
# # STEP 8: DIAGNOSTIC PLOTS
# # ============================================================
#
# # Plot 1: Actual vs Predicted
# fig, axes = plt.subplots(1, 2, figsize=(16, 6))
# fig.suptitle("XGBoost Model Diagnostics", fontsize=14, fontweight="bold")
#
# residuals = y_test - xgb_pred
#
# sns.scatterplot(x=y_test, y=xgb_pred, alpha=0.5, ax=axes[0], color="teal")
# axes[0].plot([y_test.min(), y_test.max()],
#              [y_test.min(), y_test.max()], "--r", lw=2)
# axes[0].set_title("Actual vs Predicted Salary (EGP)")
# axes[0].set_xlabel("Actual")
# axes[0].set_ylabel("Predicted")
#
# sns.scatterplot(x=xgb_pred, y=residuals, alpha=0.5, ax=axes[1], color="coral")
# axes[1].axhline(0, color="black", linestyle="-fjob-")
# axes[1].set_title("Residuals — Are errors random?")
# axes[1].set_xlabel("Predicted Salary")
# axes[1].set_ylabel("Error (Actual - Predicted)")
#
# plt.tight_layout()
# plt.savefig("model_diagnostics.png", dpi=150, bbox_inches="tight")
# # plt.show()
#
# # Plot 2: Feature Importance (Top 15)
# feat_imp = pd.Series(
#     best_model.feature_importances_,
#     index=X.columns
# ).sort_values(ascending=False).head(15).sort_values()
#
# plt.figure(figsize=(10, 7))
# feat_imp.plot(kind="barh", color="skyblue")
# plt.title("Top 15 Salary Drivers — XGBoost", fontsize=13, fontweight="bold")
# plt.xlabel("Importance Score")
# plt.tight_layout()
# plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
# # plt.show()
#
# print("\nDone! Plots saved.")



results = pd.DataFrame(
    [m_lr, m_ridge, m_xgb],
    index=["Linear Regression", "Ridge Regression", "XGBoost"]
)

print(results)