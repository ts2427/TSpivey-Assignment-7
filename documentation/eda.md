# Exploratory Data Analysis Documentation (eda.py)

## Overview
Statistical analysis of data breach patterns to identify relationships between variables and inform business decision-making.

## Purpose
Performs quantitative analysis to answer key research questions:
1. Is there a relationship between total individuals affected and state residents affected?
2. Do different organization types experience different types of breaches?

## Analyses Performed

### 1. Correlation Analysis
**Variables**: total_affected vs residents_affected  
**Methods**: 
- Pearson correlation (linear relationship)
- Spearman correlation (monotonic relationship)

**Results**:
- **Pearson**: r = 0.3150, p < 0.001
  - Moderate positive linear relationship
  - Statistically significant
- **Spearman**: rho = 0.5167, p < 0.001
  - Stronger monotonic relationship
  - Statistically significant
- **Sample size**: 11,555 valid pairs

**Interpretation**:
The difference between Pearson (0.32) and Spearman (0.52) correlations suggests:
- Non-linear relationship between variables
- Presence of outliers affecting linear correlation
- Rank-order relationship stronger than raw value relationship
- Common in breach data where extreme outliers (massive breaches) distort linear measures

### 2. Chi-Squared Test of Independence
**Variables**: organization_type vs breach_type  
**Null Hypothesis**: Organization type and breach type are independent

**Results**:
- **χ²** = 5069.93
- **p-value** < 0.001
- **Degrees of freedom** = 42
- **Sample size** = 35,378
- **Conclusion**: Reject null hypothesis - strong evidence of relationship

**Key Findings**:
Different organization types face significantly different breach threats:

- **Healthcare (MED)**:
  - DISC breaches: 1,243 observed vs 869 expected (+43%)
  - Organizations struggle with unintended disclosures

- **Financial Services (BSF)**:
  - PHYS breaches: 1,489 observed vs 553 expected (+169%)
  - Physical document security is major vulnerability

- **Retail (BSR)**:
  - CARD breaches: 45 observed vs 9 expected (+400%)
  - POS systems targeted for payment card fraud

- **Business/Other (BSO)**:
  - HACK breaches: 9,542 observed vs 8,272 expected (+15%)
  - Largest absolute number of cyber attacks

## Database Tables Created

The script creates four tables in databreach.db:

1. **correlation_results** (2 rows)
   - Stores Pearson and Spearman correlation coefficients
   - Includes p-values and sample sizes
   - Boolean flag for statistical significance

2. **chi_squared_summary** (1 row)
   - Test statistic and p-value
   - Degrees of freedom
   - Sample size and significance flag

3. **chi_squared_observed** (7 rows)
   - Contingency table of observed frequencies
   - Organization types (rows) × Breach types (columns)

4. **chi_squared_expected** (7 rows)
   - Expected frequencies under independence assumption
   - Same structure as observed table

## Business Implications

### Security Investment Priorities
- **Healthcare**: Focus on access controls and disclosure prevention
- **Financial**: Strengthen physical document security protocols
- **Retail**: Enhanced POS security and card fraud detection
- **Technology**: Advanced cyber threat detection systems

### Risk Assessment
- Different industries require different security strategies
- One-size-fits-all approach to breach prevention is ineffective
- Resource allocation should be industry-specific

### Regulatory Focus
- Healthcare DISC breaches suggest compliance challenges
- Financial PHYS breaches indicate need for document handling standards
- Retail CARD breaches require payment security regulations

## Code Location
**Script**: `eda.py`  
**Dependencies**: pandas, scipy, sqlite3  
**Execution**: Run via `python eda.py` or `python run_all.py`

## Statistical Methodology

### Data Preparation
- Numeric conversion with error coercion for invalid values
- Removal of rows with missing values in analysis variables
- No imputation performed (conservative approach)

### Significance Level
- α = 0.05 (95% confidence level)
- All reported p-values are two-tailed

### Assumptions
- **Correlation**: Assumes continuous variables (validated)
- **Chi-squared**: Minimum expected frequency ≥ 5 (validated - all cells meet requirement)

## Limitations
- Correlations show association, not causation
- Chi-squared test doesn't indicate direction or strength of specific relationships
- Analysis limited to reported breaches (reporting bias possible)
- Some breach type classifications may be ambiguous

## Next Steps
These statistical findings inform:
1. Feature selection for predictive modeling (Assignment 6)
2. Dashboard design priorities (Assignment 7)
3. Business recommendations and strategic decisions