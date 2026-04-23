import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

print("--- Step 1: Loading Data ---")
# Load the cleaned training data
df = pd.read_csv("wuzzuf_cleaned_training_data.csv")

# Fill any missing text values with empty strings to avoid errors
df['Title'] = df['Title'].fillna("")
df['Skills'] = df['Skills'].fillna("")
df['Min_Experience_Years'] = df['Min_Experience_Years'].fillna(0)

# Define our Features (X) and Target (y)
X = df[['Title', 'Skills', 'Min_Experience_Years']]
y = df['Avg_Salary']

print(f"Total training samples: {len(X)}")

# --- Step 2: Feature Engineering (Text to Numbers) ---
# We use ColumnTransformer to apply TF-IDF to text columns and leave Experience as a number
preprocessor = ColumnTransformer(
    transformers=[
        ('title_tfidf', TfidfVectorizer(max_features=100, stop_words='english'), 'Title'),
        ('skills_tfidf', TfidfVectorizer(max_features=100, stop_words='english'), 'Skills')
    ],
    remainder='passthrough' # Keep 'Min_Experience_Years' without changing it
)

# --- Step 3: Model Building (Pipelines) ---
# According to the project report, we will test Random Forest and Linear Regression
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Split the data into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 4: Training and Evaluation ---
print("\n--- Step 4: Model Evaluation ---")

# Train and test Random Forest
rf_pipeline.fit(X_train, y_train)
rf_predictions = rf_pipeline.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

print("🎯 Random Forest Performance:")
print(f"Mean Absolute Error (MAE): {rf_mae:.2f} EGP")
print(f"R-squared (R2): {rf_r2:.2f}")

# Train and test Linear Regression
lr_pipeline.fit(X_train, y_train)
lr_predictions = lr_pipeline.predict(X_test)
lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_r2 = r2_score(y_test, lr_predictions)

print("\n📉 Linear Regression Performance:")
print(f"Mean Absolute Error (MAE): {lr_mae:.2f} EGP")
print(f"R-squared (R2): {lr_r2:.2f}")

# --- Step 5: Save the Best Model ---
# Random Forest usually performs better with non-linear real-world data
best_model_filename = 'salary_prediction_model.pkl'
joblib.dump(rf_pipeline, best_model_filename)
print(f"\n✅ Best model successfully saved as '{best_model_filename}' for Dashboard deployment!")

# --- Bonus: Let's predict a Confidential Job! ---
print("\n--- Magic Time: Predicting a Confidential Salary ---")
df_confidential = pd.read_csv("wuzzuf_confidential_jobs.csv")

# 🌟 FIX: Clean and prepare the confidential data exactly like the training data
df_confidential['Title'] = df_confidential['Title'].fillna("")
df_confidential['Skills'] = df_confidential['Skills'].fillna("")
# Extract the first number from the Experience string (e.g., "3 To 7 Years" becomes 3)
df_confidential['Min_Experience_Years'] = df_confidential['Experience'].astype(str).str.extract(r'(\d+)').fillna(0).astype(int)

sample_job = df_confidential.iloc[0:1] # Take the first confidential job

print(f"Job Title: {sample_job['Title'].values[0]}")
print(f"Required Experience: {sample_job['Experience'].values[0]}")

# Predict using the loaded Random Forest pipeline
predicted_salary = rf_pipeline.predict(sample_job[['Title', 'Skills', 'Min_Experience_Years']])
print(f"🔮 AI Predicted Salary: {predicted_salary[0]:,.0f} EGP")