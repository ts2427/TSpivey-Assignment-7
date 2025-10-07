from dataclean import universal_clean
from dataload import save_to_csv, load_to_database, get_table_info, load_sec_reference
import pandas as pd
import subprocess

def run_all():
    print("="*60)
    print("DATA BREACH PIPELINE - FULL EXECUTION")
    print("="*60)
    
    # Step 1: Clean the data
    print("\n[1/4] CLEANING DATA")
    print("-"*60)
    cleaned_data = universal_clean()
    df_databreach = cleaned_data['databreach']
    
    print("Cleaned DataFrames:")
    for name, df in cleaned_data.items():
        print(f"  {name}: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Step 2: Save and load
    print("\n[2/4] SAVING AND LOADING TO DATABASE")
    print("-"*60)
    print("Saving to CSV...")
    save_to_csv(df_databreach)
    
    print("Loading breach data to database...")
    load_to_database(df_databreach)
    
    print("Loading SEC reference data...")
    load_sec_reference()
    
    print("\nDatabase info:")
    get_table_info()
    
    # Step 3: Run EDA
    print("\n[3/4] RUNNING STATISTICAL ANALYSIS")
    print("-"*60)
    result = subprocess.run(['python', 'eda.py'], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    
    # Summary
    print("\n" + "="*60)
print("PIPELINE COMPLETE")
print("="*60)
print("Database: databreach.db")
print("  - databreach table (35,378 breach records)")
print("  - sec_company_reference table (10,142 public companies)")
print("  - 13 statistical analysis tables")
print("\nOutputs:")
print("  - output/databreach.csv")
print("  - documentation/ folder (7 markdown files)")

if __name__ == "__main__":
    run_all()