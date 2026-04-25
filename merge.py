import pandas as pd
import glob
import os

def merge_and_save(folder_path="."):
    # 1. Find the CSVs
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    # Filter out the master file if it already exists to avoid "looping" data
    all_files = [f for f in all_files if "master_salary_data" not in f]
    
    if not all_files:
        print("Error: No CSV files found to merge!")
        return

    print(f"Merging: {all_files}")
    
    # 2. Stack them
    df_list = [pd.read_csv(f) for f in all_files]
    master_df = pd.concat(df_list, axis=0, ignore_index=True, sort=False)
    
    # 3. Save with ABSOLUTE path to ensure it appears
    filename = "master_salary_data.csv"
    save_path = os.path.join(os.getcwd(), filename)
    
    master_df.to_csv(save_path, index=False)
    
    print("-" * 30)
    print(f"SUCCESS: Created {filename}")
    print(f"Total rows in master file: {len(master_df)}")
    print(f"Location: {save_path}")
    print("-" * 30)

# Run the merge
merge_and_save(".")
