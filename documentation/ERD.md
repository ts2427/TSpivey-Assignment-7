Entity-Relationship Diagram (ERD)
Data Breach Analysis Database
Database Overview

Database Type: SQLite
Database File: databreach.db
Total Tables: 5 (1 main table + 4 analytical tables)
Relationships: Derived tables (no foreign key constraints)


Table Structures
1. databreach (Main Table)
Purpose: Primary storage of cleaned breach notification data
Column NameData TypeConstraintsNotesidTEXTPRIMARY KEY, NOT NULLUUID formatorg_nameTEXTOriginal organization namenormalized_org_nameTEXTStandardized name for groupingreported_dateDATETIMEISO format (YYYY-MM-DD)breach_dateDATETIMEISO format (YYYY-MM-DD)end_breach_dateDATETIMEISO format (YYYY-MM-DD)incident_detailsTEXTFree text descriptioninformation_affectedTEXTTypes of data compromisedorganization_typeTEXTCHECK(organization_type IN ('BSF','BSO','BSR','EDU','GOV','MED','NGO'))Categoricalbreach_typeTEXTNOT NULL, CHECK(breach_type IN ('CARD','DISC','HACK','INSD','PHYS','PORT','STAT'))Categoricalgroup_org_breach_typeTEXTGrouped breach classificationgroup_org_typeTEXTGrouped org classificationtotal_affectedREALCHECK(total_affected >= 0)Number of individualsresidents_affectedREALCHECK(residents_affected >= 0)State residents affectedbreach_location_streetTEXTPhysical addressbreach_location_cityTEXTCity namebreach_location_stateTEXT2-letter state codebreach_location_zipTEXTPostal codebreach_location_countryTEXTCountry nametagsTEXTComma-separated tags
Row Count: 35,378
Indexes: Primary key on id

2. correlation_results
Purpose: Stores Pearson and Spearman correlation analysis results
Column NameData TypeConstraintsNotestest_typeTEXTNOT NULL'Pearson' or 'Spearman'variable_1TEXTNOT NULLFirst variable namevariable_2TEXTNOT NULLSecond variable namecoefficientREALNOT NULLCorrelation coefficientp_valueREALNOT NULLStatistical significancesample_sizeINTEGERNOT NULLNumber of valid pairssignificant_at_0.05BOOLEANNOT NULLTRUE if p < 0.05
Row Count: 2 (one row per test type)
Source: Derived from databreach.total_affected and databreach.residents_affected
Created by: eda.py

3. chi_squared_summary
Purpose: Stores chi-squared test statistics
Column NameData TypeConstraintsNotestest_typeTEXTNOT NULLAlways 'Chi-Squared'variable_1TEXTNOT NULLFirst categorical variablevariable_2TEXTNOT NULLSecond categorical variablechi_squared_statisticREALNOT NULLChi-squared test statisticp_valueREALNOT NULLStatistical significancedegrees_of_freedomINTEGERNOT NULLdf for the testsample_sizeINTEGERNOT NULLTotal observationssignificant_at_0.05BOOLEANNOT NULLTRUE if p < 0.05
Row Count: 1
Source: Derived from databreach.organization_type and databreach.breach_type
Created by: eda.py

4. chi_squared_observed
Purpose: Contingency table of observed frequencies
Column NameData TypeConstraintsNotesorganization_typeTEXTNOT NULLRow headers (BSF, BSO, BSR, EDU, GOV, MED, NGO)CARDINTEGERCount of CARD breachesDISCINTEGERCount of DISC breachesHACKINTEGERCount of HACK breachesINSDINTEGERCount of INSD breachesPHYSINTEGERCount of PHYS breachesPORTINTEGERCount of PORT breachesSTATINTEGERCount of STAT breaches
Row Count: 7 (one per organization type, UNKN excluded)
Source: Cross-tabulation of databreach table
Created by: eda.py

5. chi_squared_expected
Purpose: Expected frequencies under independence assumption
Column NameData TypeConstraintsNotesorganization_typeTEXTNOT NULLRow headers (BSF, BSO, BSR, EDU, GOV, MED, NGO)CARDREALExpected count for CARDDISCREALExpected count for DISCHACKREALExpected count for HACKINSDREALExpected count for INSDPHYSREALExpected count for PHYSPORTREALExpected count for PORTSTATREALExpected count for STAT
Row Count: 7 (one per organization type)
Source: Chi-squared test calculation from scipy.stats
Created by: eda.py

Table Relationships
databreach (35,378 rows)
    │
    ├──> correlation_results (2 rows)
    │    Analysis: total_affected vs residents_affected
    │
    ├──> chi_squared_summary (1 row)
    │    Analysis: organization_type vs breach_type
    │
    ├──> chi_squared_observed (7 rows)
    │    Cross-tabulation of actual frequencies
    │
    └──> chi_squared_expected (7 rows)
         Cross-tabulation of expected frequencies
Relationship Type: All analytical tables are derived from the databreach table
Foreign Keys: None (tables are independent; linked conceptually, not relationally)
Referential Integrity: Not enforced by database constraints

Database Schema Diagram
┌─────────────────────────────────────┐
│          databreach                 │
│  (Main Breach Records)              │
├─────────────────────────────────────┤
│ PK: id                              │
│ • org_name                          │
│ • normalized_org_name               │
│ • reported_date                     │
│ • breach_date                       │
│ • end_breach_date                   │
│ • incident_details                  │
│ • information_affected              │
│ • organization_type (7 categories)  │
│ • breach_type (7 categories)        │
│ • group_org_breach_type             │
│ • group_org_type                    │
│ • total_affected                    │
│ • residents_affected                │
│ • breach_location_* (5 fields)      │
│ • tags                              │
│                                     │
│ Rows: 35,378                        │
└─────────────────────────────────────┘
              │
              │ (source data for)
              │
    ┌─────────┴─────────┐
    │                   │
    ▼                   ▼
┌──────────────┐  ┌──────────────────┐
│correlation_  │  │chi_squared_      │
│results       │  │summary           │
├──────────────┤  ├──────────────────┤
│test_type     │  │test_type         │
│variable_1    │  │variable_1        │
│variable_2    │  │variable_2        │
│coefficient   │  │chi_squared_stat  │
│p_value       │  │p_value           │
│sample_size   │  │dof               │
│significant   │  │sample_size       │
│              │  │significant       │
│Rows: 2       │  │                  │
└──────────────┘  │Rows: 1           │
                  └──────────────────┘
                          │
                  ┌───────┴────────┐
                  │                │
                  ▼                ▼
          ┌──────────────┐  ┌──────────────┐
          │chi_squared_  │  │chi_squared_  │
          │observed      │  │expected      │
          ├──────────────┤  ├──────────────┤
          │org_type      │  │org_type      │
          │CARD          │  │CARD          │
          │DISC          │  │DISC          │
          │HACK          │  │HACK          │
          │INSD          │  │INSD          │
          │PHYS          │  │PHYS          │
          │PORT          │  │PORT          │
          │STAT          │  │STAT          │
          │              │  │              │
          │Rows: 7       │  │Rows: 7       │
          └──────────────┘  └──────────────┘

Normalization Assessment
Current Form: 2NF (Second Normal Form)

1NF: ✓ All columns contain atomic values
2NF: ✓ No partial dependencies (single column primary key)
3NF: ✗ Some transitive dependencies exist (e.g., group_* fields depend on normalized_org_name)

Design Rationale:

Intentionally denormalized for analytical queries
Optimized for read-heavy workload
Eliminates need for joins in most queries
Trade-off: Some redundancy for better query performance


Indexing Strategy
Current Indexes:

Primary key index on databreach.id (automatic)

Recommended Additional Indexes (for future optimization):
sqlCREATE INDEX idx_org_type ON databreach(organization_type);
CREATE INDEX idx_breach_type ON databreach(breach_type);
CREATE INDEX idx_breach_date ON databreach(breach_date);
CREATE INDEX idx_normalized_org ON databreach(normalized_org_name);