# Data Cleaning Documentation (dataclean.py)
Overview
The dataclean.py script processes the raw Data Breach Chronology Excel file and performs essential cleaning operations to prepare the data for analysis.
Functions
add_databreach_data(filename)
Purpose: Reads and cleans the Data Breach Excel file
Input:

filename (str)++: Path to the Excel file (e.g., 'Data_Breach_Chronology.xlsx')

Process:

Column Selection: Reduces dataset to 20 essential columns from original file
Filtering: Removes records where breach_type == 'UNKN' (unknown breach types)
Date Conversion: Converts three date columns to proper datetime format
Numeric Conversion: Converts affected counts to numeric types

Output:

Cleaned pandas DataFrame


Columns Retained
The following 20 columns are kept from the original dataset:
ColumnTypeDescriptionidStringUnique identifierorg_nameStringOriginal organization namereported_dateDatetimeWhen breach was reportedbreach_dateDatetimeWhen breach occurredend_breach_dateDatetimeWhen breach endedincident_detailsStringDescription of the incidentinformation_affectedStringTypes of data compromisedorganization_typeStringBSF, BSO, BSR, EDU, GOV, MED, NGObreach_typeStringCARD, DISC, HACK, INSD, PHYS, PORT, STATnormalized_org_nameStringStandardized company namegroup_org_breach_typeStringGrouped breach type categorygroup_org_typeStringGrouped organization categorytotal_affectedNumericTotal individuals affectedresidents_affectedNumericResidents affectedbreach_location_streetStringStreet addressbreach_location_cityStringCitybreach_location_stateStringState/Provincebreach_location_zipStringZIP/Postal codebreach_location_countryStringCountrytagsStringAdditional categorization tags

Data Transformations
1. Breach Type Filtering

Action: Removes all records where breach_type == 'UNKN'
Rationale: Unknown breach types cannot be meaningfully analyzed
Impact: Reduces dataset size but improves data quality

2. Date Standardization

Columns affected: reported_date, breach_date, end_breach_date
Method: pd.to_datetime() with errors='coerce'
Result: Invalid dates converted to NaT (Not a Time)

3. Numeric Conversion

Columns affected: total_affected, residents_affected
Method: pd.to_numeric() with errors='coerce'
Result: Invalid numbers converted to NaN


Data Quality Handling
Missing Data Strategy

Coercion approach: Invalid values are converted to NaT or NaN rather than causing errors
Preservation: All records retained even with missing values in some fields
Analysis impact: Statistical analyses will automatically exclude missing values

Column Reduction Rationale

Original dataset contains 37+ columns
Reduced to 20 core columns needed for analysis
Eliminated metadata and explanation columns (e.g., *_explanation fields)
Removed system fields (e.g., created_at, updated_at, source)


Usage
universal_clean()
Purpose: Main function to execute all cleaning operations
Returns: Dictionary with cleaned dataframes
python{
    'databreach': df_databreach_clean
}
Example:
pythonfrom dataclean import universal_clean

cleaned_data = universal_clean()
databreach_df = cleaned_data['databreach']

Output Characteristics
After cleaning, the dataset:

Contains only analyzable breach types (excludes UNKN)
Has properly typed date and numeric columns
Retains all essential fields for statistical analysis
Is ready for database loading via dataload.py


Next Steps
After cleaning, the data is loaded into SQLite database using dataload.py.