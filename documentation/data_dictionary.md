# Data Dictionary - databreach Table

## Table Overview

- **Table Name:** databreach
- **Primary Key:** id
- **Total Columns:** 20
- **Total Rows:** 35,378 (after cleaning)
- **Source:** Privacy Rights Clearinghouse Data Breach Chronology v2.1

## Column Specifications

| Column Name | Data Type | Description | Nullable | Constraints | Example Value |
|-------------|-----------|-------------|----------|-------------|---------------|
| id | TEXT | Unique identifier (UUID) for each breach record | No | PRIMARY KEY, UNIQUE | "550e8400-e29b-41d4-a716-446655440000" |
| org_name | TEXT | Original organization name as reported in breach notification | Yes | - | "Global Tech Solutions, Inc." |
| normalized_org_name | TEXT | Standardized organization name for tracking across multiple reports | Yes | - | "Global Tech Solutions" |
| reported_date | DATETIME | Date when breach was reported to government agency | Yes | Format: YYYY-MM-DD | "2024-03-15" |
| breach_date | DATETIME | Date when breach occurred or was discovered | Yes | Format: YYYY-MM-DD | "2024-01-15" |
| end_breach_date | DATETIME | Date when breach was contained/ended (if applicable) | Yes | Format: YYYY-MM-DD | "2024-01-31" |
| incident_details | TEXT | Detailed description of the breach event and circumstances | Yes | - | "Unauthorized access to employee email account containing patient information" |
| information_affected | TEXT | Types of personal information compromised in the breach | Yes | - | "Names, Social Security numbers, medical records" |
| organization_type | TEXT | Classification of the breached organization | Yes | Valid values: BSF, BSO, BSR, EDU, GOV, MED, NGO | "MED" |
| breach_type | TEXT | Primary method or nature of the breach | No | Valid values: CARD, DISC, HACK, INSD, PHYS, PORT, STAT (UNKN excluded during cleaning) | "HACK" |
| group_org_breach_type | TEXT | Grouped/consolidated breach type for related incidents | Yes | Same values as breach_type | "HACK" |
| group_org_type | TEXT | Grouped/consolidated organization type for related incidents | Yes | Same values as organization_type | "MED" |
| total_affected | REAL | Total number of individuals impacted across all jurisdictions | Yes | Must be >= 0 | 15000.0 |
| residents_affected | REAL | Number of state residents affected (varies by reporting state) | Yes | Must be >= 0 | 5000.0 |
| breach_location_street | TEXT | Street address where breach occurred | Yes | - | "5550 Peachtree Parkway, Suite 500" |
| breach_location_city | TEXT | City where breach occurred | Yes | - | "Peachtree Corners" |
| breach_location_state | TEXT | State/province where breach occurred | Yes | 2-letter state code | "GA" |
| breach_location_zip | TEXT | ZIP/postal code where breach occurred | Yes | - | "30092" |
| breach_location_country | TEXT | Country where breach occurred | Yes | - | "United States" |
| tags | TEXT | Additional categorization tags for the breach | Yes | Comma-separated values | "sensitive-personal-information,finance" |

## Field Value Definitions

### organization_type Values

| Code | Full Name | Description |
|------|-----------|-------------|
| BSF | Financial Services Business | Banks, credit unions, investment firms, insurance carriers (excluding health) |
| BSO | Other Business | Technology companies, manufacturers, utilities, professional services |
| BSR | Retail Business | Physical and online retail merchants |
| EDU | Educational Institutions | Schools, universities, educational services |
| GOV | Government and Military | Public administration, government agencies |
| MED | Healthcare Providers | Hospitals, clinics, HIPAA-covered entities |
| NGO | Nonprofits | Charities, advocacy groups, religious organizations |

### breach_type Values

| Code | Full Name | Description |
|------|-----------|-------------|
| CARD | Card Compromise | Physical payment card compromises (skimming, POS tampering) |
| DISC | Disclosure | Unintended disclosures (misconfiguration, accidents) |
| HACK | Hacking/Cyber Attack | External cyber attacks (malware, ransomware, network intrusions) |
| INSD | Insider Threat | Internal threats from authorized users |
| PHYS | Physical Theft/Loss | Physical document theft or loss |
| PORT | Portable Device | Portable device breaches (laptops, phones, tablets) |
| STAT | Stationary Device | Stationary device breaches (desktops, servers) |

## Data Quality Notes

**Missing Values:**
- High missingness: breach_location_street (~70% missing)
- Moderate missingness: end_breach_date (~40% missing)
- Low missingness: reported_date, breach_date (<5% missing)
- No missing values: id, breach_type (filtered during cleaning)

**Data Transformations Applied:**
- Date fields converted from text to datetime objects
- Numeric fields (total_affected, residents_affected) converted from text to float
- Invalid values coerced to NULL (NaT for dates, NaN for numbers)
- Records with breach_type = 'UNKN' removed during cleaning

**Data Validation Rules:**
- All dates must be between 2005-01-01 and 2025-12-31
- total_affected should be >= residents_affected (when both present)
- breach_date should be <= reported_date (when both present)
- organization_type and breach_type must match valid code lists

## Usage Examples

### Query breaches by type
```sql
SELECT organization_type, COUNT(*) as breach_count
FROM databreach
GROUP BY organization_type
ORDER BY breach_count DESC;
```

### Find large breaches
```sql
SELECT org_name, breach_date, total_affected
FROM databreach
WHERE total_affected > 100000
ORDER BY total_affected DESC;
```

### Filter by date range
```sql
SELECT *
FROM databreach
WHERE breach_date BETWEEN '2023-01-01' AND '2023-12-31';
```
```

---

## FILE 5: analysis_methodology.md (PARTIAL - KEY FORMULA SECTIONS)

For this file, find these sections and replace the formula blocks:

**FIND THIS SECTION:**
```
**Formula Reference:**
```
Mean (Î¼) = Î£x / n
```

**REPLACE WITH:**
```
**Formula Reference:**
```
Mean (μ) = Σx / n
Std Dev (σ) = √[Σ(x - μ)² / (n-1)]
Skewness = [n / (n-1)(n-2)] × Σ[(x - μ)/σ]³
Kurtosis = [n(n+1) / (n-1)(n-2)(n-3)] × Σ[(x - μ)/σ]⁴ - [3(n-1)² / (n-2)(n-3)]
IQR = Q3 - Q1
CV = (σ / μ) × 100%
```
```

**FIND THIS SECTION:**
```
**Formula:**
```
r = Î£[(x - xÌ„)(y - È³)] / âˆš[Î£(x - xÌ„)Â² Ã— Î£(y - È³)Â²]
```

**REPLACE WITH:**
```
**Formula:**
```
r = Σ[(x - x̄)(y - ȳ)] / √[Σ(x - x̄)² × Σ(y - ȳ)²]
```