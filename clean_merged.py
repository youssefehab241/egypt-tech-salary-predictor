import pandas as pd
import numpy as np

def clean_ml_data(file_path):
    # Load data
    df = pd.read_csv(file_path)

    # ---------------------------------------------------------
    # 1. SCHEMA ALIGNMENT (Coalescing fragmented columns)
    # ---------------------------------------------------------
    # Combine job titles and lowercase them
    df['job_title_clean'] = df['job_title'].fillna(df['Title']).str.lower().str.strip()

    # Combine the different salary columns into one target variable
    df['salary_target'] = df['salary_mid'].fillna(df['average_salary']).fillna(df['Avg_Salary'])

    # Combine experience columns
    df['experience_years_clean'] = df['experience_years'].fillna(df['Min_Experience_Years'])

    # Combine the three different location columns and standardize casing
    df['location_clean'] = df['city'].fillna(df['Clean_Location']).fillna(df['location']).str.title().str.strip()

    # ---------------------------------------------------------
    # 2. STANDARDIZING CATEGORICAL TEXT
    # ---------------------------------------------------------
    # Fix typos in Work Mode
    def standardize_work_mode(mode):
        if pd.isna(mode): return 'Unknown'
        m = str(mode).lower()
        if 'hyb' in m: return 'Hybrid'
        if 'remot' in m: return 'Remote'
        if 'site' in m or 'office' in m: return 'On-Site'
        return 'Unknown'

    df['work_mode_clean'] = df['work_mode'].apply(standardize_work_mode)

    # Group Locations (Optional: Groups Giza and Cairo into 'Greater Cairo')
    def standardize_location(loc):
        if pd.isna(loc): return 'Unknown'
        if 'Cairo' in loc or 'Giza' in loc: return 'Greater Cairo'
        if 'Alex' in loc: return 'Alexandria'
        return loc

    df['location_clean'] = df['location_clean'].apply(standardize_location)

    # ---------------------------------------------------------
    # 3. MISSING VALUE IMPUTATION
    # ---------------------------------------------------------
    # A model cannot train if the target variable (Salary) is missing. Drop those rows.
    df = df.dropna(subset=['salary_target'])

    # Impute missing experience with the dataset's Median
    median_exp = df['experience_years_clean'].median()
    df['experience_years_clean'] = df['experience_years_clean'].fillna(median_exp)

    # Extract Seniority Level (Use the explicit column, or guess from Job Title)
    def extract_seniority(row):
        val = str(row['position_level_raw']).lower()
        title = str(row['job_title_clean']).lower()

        # Check raw level column first
        if val != 'nan':
            if 'senior' in val or 'sr' in val: return 'Senior'
            if 'junior' in val or 'jr' in val: return 'Junior'
            if 'mid' in val: return 'Mid-Level'
            if 'fresh' in val: return 'Entry-Level'
            if 'lead' in val or 'manager' in val: return 'Lead/Manager'

        # Fallback to guessing from the job title
        if 'senior' in title or 'sr' in title: return 'Senior'
        if 'junior' in title or 'jr' in title: return 'Junior'
        if 'lead' in title or 'manager' in title: return 'Lead/Manager'

        return 'Unknown'

    df['level_clean'] = df.apply(extract_seniority, axis=1)

    # ---------------------------------------------------------
    # 4. FINAL EXPORT
    # ---------------------------------------------------------
    # Select ONLY the clean, finalized features for the ML model
    ml_columns = [
        'job_title_clean',
        'salary_target',
        'experience_years_clean',
        'location_clean',
        'work_mode_clean',
        'level_clean'
    ]

    ml_df = df[ml_columns]

    # Save the cleaned dataset
    ml_df.to_csv('ml_ready_salary_data.csv', index=False)
    print(f"Data cleaned and saved! Final usable rows: {len(ml_df)}")
    return ml_df

# Execute the script
clean_data = clean_ml_data('master_salary_data.csv')
