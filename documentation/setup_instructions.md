Setup and Run Instructions
Project: Data Breach Analysis Database
Overview
This project implements an ETL (Extract, Transform, Load) pipeline for analyzing data breach notifications from Privacy Rights Clearinghouse. The pipeline cleans raw breach data, loads it into a SQLite database, and performs statistical analyses.

Prerequisites
Required Software

Python 3.8 or higher
pip (Python package manager)

Required Python Libraries
bashpip install pandas openpyxl sqlite3 scipy
Library Versions Used:

pandas >= 1.5.0
openpyxl >= 3.0.0 (for Excel file reading)
scipy >= 1.9.0 (for statistical tests)
sqlite3 (included with Python)


Project Structure
DATABREACH/
├── Data_Breach_Chronology.xlsx          # Raw data file
├── CIK_NAME_TICKER_EXCHANGE.csv         # SEC reference data
├── company_tickers.csv                  # Manual ticker matches
├── databreach.db                        # SQLite database (created by pipeline)
├── dataclean.py                         # Data cleaning script
├── dataload.py                          # Database loading script
├── eda.py                               # Exploratory data analysis script
├── output/
│   └── databreach.csv                   # CSV export (created by pipeline)
└── documentation/
    ├── cleaning.md                      # Cleaning process documentation
    ├── loading.md                       # Loading process documentation
    ├── sources.md                       # Data sources documentation
    ├── eda.md                          # EDA documentation
    ├── database_schema.md              # ERD and schema details
    └── data_dictionary.md              # Column specifications

Installation Instructions
Step 1: Clone or Download Project
bash# If using Git
git clone [your-repository-url]
cd DATABREACH

# Or download and extract ZIP file
Step 2: Install Dependencies
bash# Install required Python packages
pip install pandas openpyxl scipy

# Verify installation
python -c "import pandas, openpyxl, scipy; print('All dependencies installed successfully')"
Step 3: Verify Data Files
Ensure the following files are in the project directory:

Data_Breach_Chronology.xlsx
CIK_NAME_TICKER_EXCHANGE.csv (optional, for enrichment)
company_tickers.csv (optional, for enrichment)


Running the Pipeline
Option 1: Run Complete Pipeline (Recommended)
Create a file called run_pipeline.py:
pythonfrom dataclean import universal_clean
from dataload import save_to_csv, load_to_database, get_table_info

print("="*60)
print("DATA BREACH ANALYSIS PIPELINE")
print("="*60)

# Step 1: Clean the data
print("\n[1/3] Cleaning data...")
cleaned_data = universal_clean()
df_databreach = cleaned_data['databreach']
print(f"✓ Cleaned {len(df_databreach):,} records")

# Step 2: Save to CSV
print("\n[2/3] Saving to CSV...")
save_to_csv(df_databreach)
print("✓ CSV saved to output/databreach.csv")

# Step 3: Load to database
print("\n[3/3] Loading to database...")
load_to_database(df_databreach)
print("✓ Database created: databreach.db")

# Verify
print("\n" + "="*60)
print("PIPELINE COMPLETE")
print("="*60)
get_table_info()
Then run:
bashpython run_pipeline.py
Option 2: Run Scripts Individually
Step 1: Clean Data
pythonfrom dataclean import universal_clean

cleaned_data = universal_clean()
df_databreach = cleaned_data['databreach']
print(f"Cleaned {len(df_databreach)} records")
Step 2: Load to Database
pythonfrom dataload import load_to_database, save_to_csv

# Save CSV backup
save_to_csv(df_databreach)

# Load to SQLite
load_to_database(df_databreach)
Step 3: Run Statistical Analysis
bashpython eda.py

Expected Output
After Successful Run
Files Created:

databreach.db - SQLite database
output/databreach.csv - CSV export

Database Tables Created:

databreach - Main table (35,378 rows, 20 columns)
correlation_results - Correlation analysis (2 rows)
chi_squared_summary - Chi-squared test results (1 row)
chi_squared_observed - Observed frequencies (7 rows)
chi_squared_expected - Expected frequencies (7 rows)

Console Output:
====================================================
DATA BREACH ANALYSIS PIPELINE
====================================================

[1/3] Cleaning data...
✓ Cleaned 35,378 records

[2/3] Saving to CSV...
Saved databreach to output/databreach.csv
✓ CSV saved to output/databreach.csv

[3/3] Loading to database...
Loaded databreach into database table
Database saved as databreach.db
✓ Database created: databreach.db

====================================================
PIPELINE COMPLETE
====================================================
databreach table: 35378 rows

Verifying the Database
Method 1: Using Python
pythonimport sqlite3
import pandas as pd

conn = sqlite3.connect('databreach.db')

# Check table exists
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("Tables:", cursor.fetchall())

# Load and preview data
df = pd.read_sql_query("SELECT * FROM databreach LIMIT 5", conn)
print(df)

conn.close()
Method 2: Using DB Browser for SQLite

Download DB Browser for SQLite (free): https://sqlitebrowser.org/
Open databreach.db
Browse tables and data visually

Method 3: Using Command Line
bashsqlite3 databreach.db "SELECT COUNT(*) FROM databreach;"
# Should output: 35378

Running Statistical Analysis
After the database is created, run exploratory data analysis:
bashpython eda.py
Output:
====================================================
DATA LOADED
====================================================
Total rows: 35,378
Total columns: 20

====================================================
CORRELATION ANALYSIS
====================================================
Valid pairs for correlation: 11,555

Pearson: r=0.3150, p-value=0.000000
Spearman: rho=0.5167, p-value=0.000000
✓ Correlation results saved to database

====================================================
CHI-SQUARED TEST: Organization Type vs Breach Type
====================================================
Valid observations: 35,378

Chi-squared statistic: 5069.9339
P-value: 0.000000
Degrees of freedom: 42
Result: SIGNIFICANT (α=0.05)
→ There IS a significant relationship between organization type and breach type
✓ Chi-squared summary saved to database
✓ Observed frequencies saved to database
✓ Expected frequencies saved to database

====================================================
EDA COMPLETE - ALL RESULTS SAVED!
====================================================

Troubleshooting
Error: "No module named 'pandas'"
Solution: Install pandas
bashpip install pandas
Error: "File not found: Data_Breach_Chronology.xlsx"
Solution: Ensure the Excel file is in the same directory as the scripts, or provide the full path in dataclean.py
Error: "Database is locked"
Solution: Close any other programs accessing databreach.db (like DB Browser) and try again
Error: "openpyxl is required for reading Excel files"
Solution: Install openpyxl
bashpip install openpyxl
Warning: "errors='coerce' produced NaN values"
This is expected: Invalid dates and numbers are converted to NaN/NaT during cleaning. This is intentional behavior.

Data Quality Validation
To verify data quality after loading:
pythonfrom dataload import load_from_database
import pandas as pd

df = load_from_database()

# Check record count
print(f"Total records: {len(df):,}")

# Check for required columns
required_cols = ['id', 'org_name', 'breach_type', 'organization_type']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"⚠ Missing columns: {missing_cols}")
else:
    print("✓ All required columns present")

# Check data types
print("\nData types:")
print(df[['reported_date', 'breach_date', 'total_affected']].dtypes)

# Check for null values in critical fields
print("\nNull counts in key fields:")
print(df[['id', 'breach_type', 'organization_type']].isnull().sum())

Next Steps
After successful pipeline execution:

Review Documentation: Read files in documentation/ folder
Explore Database: Use DB Browser or SQL queries to examine data
Analyze Results: Review statistical outputs in analytical tables
Extend Analysis: Add additional queries or visualizations as needed


Support and Contact
For questions or issues:

Review documentation in documentation/ folder
Check assignment requirements
Contact: [Your email/contact information]


File Locations Quick Reference
ItemLocationRaw dataData_Breach_Chronology.xlsxDatabasedatabreach.dbCSV exportoutput/databreach.csvCleaning scriptdataclean.pyLoading scriptdataload.pyAnalysis scripteda.pyDocumentationdocumentation/*.md