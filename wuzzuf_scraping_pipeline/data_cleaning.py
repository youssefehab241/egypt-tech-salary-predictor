import pandas as pd
import re
import numpy as np

# 1. Load the raw dataset
df = pd.read_csv("wuzzuf_big_data_complete.csv")
print(f"Total jobs loaded: {len(df)}")

df_train = df[df['Salary'] != 'Confidential'].copy()
df_predict = df[df['Salary'] == 'Confidential'].copy()

# 2. Clean and Parse Salary
def parse_salary(salary_str):
    try:
        salary_str = salary_str.upper()
        is_usd = 'USD' in salary_str or '$' in salary_str
        
        numbers = re.findall(r'[\d,]+', salary_str)
        numbers = [int(num.replace(',', '')) for num in numbers]
        
        if len(numbers) == 0:
            return pd.Series([np.nan, np.nan, np.nan])
        
        min_sal = numbers[0]
        max_sal = numbers[1] if len(numbers) > 1 else numbers[0]
        
        if is_usd:
            min_sal *= 50
            max_sal *= 50
            
        avg_sal = (min_sal + max_sal) / 2
        return pd.Series([min_sal, max_sal, avg_sal])
        
    except Exception as e:
        return pd.Series([np.nan, np.nan, np.nan])

df_train[['Min_Salary', 'Max_Salary', 'Avg_Salary']] = df_train['Salary'].apply(parse_salary)

# 3. Clean and Parse Experience
def parse_experience(exp_str):
    try:
        if pd.isna(exp_str) or exp_str == "Not Specified" or exp_str == "N/A":
            return 0 
        numbers = re.findall(r'\d+', str(exp_str))
        if numbers:
            return int(numbers[0])
        return 0
    except:
        return 0

df_train['Min_Experience_Years'] = df_train['Experience'].apply(parse_experience)
df_train['Clean_Location'] = df_train['Location'].apply(lambda x: str(x).split(', ')[-1] if pd.notna(x) else "Unknown")
df_train.dropna(subset=['Avg_Salary'], inplace=True)

# ==========================================
# 🌟 التعديل السحري: إزالة القيم الشاذة (Outliers) 🌟
# هنستبعد أي راتب أقل من 3000 جنيه أو أعلى من 200,000 جنيه شهرياً
# ==========================================
initial_count = len(df_train)
df_train = df_train[(df_train['Avg_Salary'] >= 3000) & (df_train['Avg_Salary'] <= 250000)]
removed_outliers = initial_count - len(df_train)
print(f"🧹 Removed {removed_outliers} outlier jobs (fake or yearly salaries).")

# 4. Select final columns and save
final_columns = ['Title', 'Company', 'Clean_Location', 'Job_Type', 'Skills', 'Min_Experience_Years', 'Min_Salary', 'Max_Salary', 'Avg_Salary']
df_ready_for_ml = df_train[final_columns]

df_ready_for_ml.to_csv("wuzzuf_cleaned_training_data.csv", index=False, encoding="utf-8-sig")
df_predict.to_csv("wuzzuf_confidential_jobs.csv", index=False, encoding="utf-8-sig")

print(f"✅ Cleaned training dataset saved! (Rows ready for ML: {len(df_ready_for_ml)})\n")