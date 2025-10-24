# Data Loading Documentation (dataload.py)

## Overview
The dataload.py script handles saving cleaned data to CSV files and loading data into a SQLite database for persistent storage and analysis.

## Functions

### save_to_csv(df_databreach, output_dir='output')

**Purpose:** Exports the cleaned databreach DataFrame to a CSV file

**Parameters:**
- `df_databreach` (DataFrame): Cleaned databreach data
- `output_dir` (str, optional): Output directory path. Default: 'output'

**Process:**
1. Creates the output directory if it doesn't exist
2. Saves DataFrame to output/databreach.csv
3. Prints confirmation message

**Output:**
- CSV file: output/databreach.csv

**Example:**
```python
save_to_csv(cleaned_df)
# Output: Saved databreach to output/databreach.csv
```

### load_to_database(df_databreach, db_name='databreach.db')

**Purpose:** Loads the databreach DataFrame into a SQLite database

**Parameters:**
- `df_databreach` (DataFrame): Cleaned databreach data
- `db_name` (str, optional): Database filename. Default: 'databreach.db'

**Process:**
1. Connects to SQLite database (creates if doesn't exist)
2. Writes DataFrame to table named 'databreach'
3. Uses if_exists='replace' to overwrite existing table
4. Closes database connection

**Database Table:** databreach

**Example:**
```python
load_to_database(cleaned_df)
# Output: 
# Loaded databreach into database table
# Database saved as databreach.db
```

### load_from_database(db_name='databreach.db')

**Purpose:** Retrieves the databreach table from the SQLite database

**Parameters:**
- `db_name` (str, optional): Database filename. Default: 'databreach.db'

**Returns:**
- pandas DataFrame containing all records from the databreach table

**Example:**
```python
df = load_from_database()
print(f"Loaded {len(df)} records")
```

### get_table_info(db_name='databreach.db')

**Purpose:** Displays row count information for the databreach table

**Parameters:**
- `db_name` (str, optional): Database filename. Default: 'databreach.db'

**Output:**
- Prints row count to console

**Example:**
```python
get_table_info()
# Output: databreach table: 35378 rows
```

## Database Schema

### Table: databreach

The database contains a single table with the following structure:

| Column | SQLite Type | Description |
|--------|------------|-------------|
| id | TEXT | Unique identifier |
| org_name | TEXT | Original organization name |
| reported_date | TEXT | When breach was reported (datetime) |
| breach_date | TEXT | When breach occurred (datetime) |
| end_breach_date | TEXT | When breach ended (datetime) |
| incident_details | TEXT | Description of the incident |
| information_affected | TEXT | Types of data compromised |
| organization_type | TEXT | Organization category |
| breach_type | TEXT | Breach method category |
| normalized_org_name | TEXT | Standardized company name |
| group_org_breach_type | TEXT | Grouped breach type |
| group_org_type | TEXT | Grouped organization type |
| total_affected | REAL | Total individuals affected |
| residents_affected | REAL | Residents affected |
| breach_location_street | TEXT | Street address |
| breach_location_city | TEXT | City |
| breach_location_state | TEXT | State/Province |
| breach_location_zip | TEXT | ZIP/Postal code |
| breach_location_country | TEXT | Country |
| tags | TEXT | Additional tags |

**Note:** SQLite stores datetime objects as TEXT. Numeric columns are stored as REAL (floating-point).

## Typical Workflow

### 1. Initial Data Load
```python
from dataclean import universal_clean
from dataload import load_to_database, save_to_csv

# Clean the data
cleaned_data = universal_clean()
df_databreach = cleaned_data['databreach']

# Save to CSV (backup/portability)
save_to_csv(df_databreach)

# Load to database (for analysis)
load_to_database(df_databreach)
```

### 2. Subsequent Analysis
```python
from dataload import load_from_database, get_table_info

# Check database status
get_table_info()

# Load data for analysis
df = load_from_database()
```

## File Outputs

**CSV Output:**
- Location: output/databreach.csv
- Purpose: Human-readable backup, data portability
- Format: Comma-separated values with header row

**Database Output:**
- Location: databreach.db (project root)
- Purpose: Efficient querying, persistent storage, relational operations
- Format: SQLite database file

## Design Decisions

### Why SQLite?
- No server required: File-based database
- Portable: Single file contains entire database
- SQL support: Enables complex queries and joins
- Python integration: Native support via sqlite3 module
- Performance: Fast for datasets of this size

### Why Also Save CSV?
- Backup: Human-readable format
- Portability: Easy to share or import into other tools
- Transparency: Reviewable in Excel, text editors
- Version control: Can be tracked in Git (if small enough)

### Table Replacement Strategy
- Uses if_exists='replace' to overwrite existing tables
- Ensures database always contains latest cleaned data
- Simplifies pipeline: no need to manually drop tables

## Error Handling

**Directory Creation:**
- save_to_csv() automatically creates the output/ directory if missing
- Uses os.makedirs() to create nested directories if needed

**Database Connections:**
- All functions properly close database connections using conn.close()
- Prevents database locks and resource leaks

## Next Steps

After loading, the data is ready for:
- Exploratory data analysis (eda.py)
- Statistical testing
- Data enrichment with ticker/CIK information