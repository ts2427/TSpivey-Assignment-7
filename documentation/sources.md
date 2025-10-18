Data Sources Documentation
Overview
This document describes all data sources used in the data breach analysis project, including their origins, update frequencies, and usage purposes.

Primary Data Sources
1. Data Breach Chronology
File: Data_Breach_Chronology.xlsx
Source: Privacy Rights Clearinghouse

Website: https://privacyrights.org
Project: Data Breach Chronology Database
Version: 2.1 (May 2025)

Description: Comprehensive database of data breach notifications reported to state and federal agencies across the United States. Since 2005, Privacy Rights Clearinghouse has tracked these breaches by combining data from government agency reports and detailed notification letters sent to affected individuals.
Coverage:

Time period: 2005 - June 2025
Geographic scope: United States only
Total records: 74,797 breach incidents (raw data before cleaning)
After cleaning: ~35,378 records (excluding UNKN breach types)

Data Collection Method:

Custom scrapers for 15 state and federal government sources
Automated monitoring of state attorney general offices
PDF downloading and archiving of breach notification letters
AI-driven processing and classification
Manual review and validation

Government Sources:

Federal: U.S. Department of Health and Human Services
State: California, Delaware, Indiana, Iowa, Maine, Maryland, Massachusetts, Montana, New Hampshire, Oregon, Texas, Vermont, Washington, Wisconsin

Update Frequency: Monthly (as of Version 2.0)
Last Updated: June 2025
Key Fields:

Organization identifiers (org_name, normalized_org_name, acceptable_names)
Breach characteristics (breach_type, incident_details)
Temporal data (reported_date, breach_date, end_breach_date)
Impact metrics (total_affected, residents_affected)
Location information (street, city, state, zip, country)
Classification (organization_type, breach_type)
Grouping fields (group_uuid, normalized organization identifiers)

Data Quality Notes:

Only includes publicly disclosed breaches reported to government agencies
Some records contain unknown breach types (UNKN) - filtered out during cleaning
Missing values present in some date and numeric fields
Standardized organization names provided via normalized_org_name
Breach grouping may not capture all related incidents or may group some unrelated ones
Classification based on information available in breach notifications

Processing Improvements (Version 2.1):

Enhanced AI models for categorization
Refactored grouping algorithms for better clustering
Improved normalization and deduplication of organization names


2. SEC Company Tickers and CIK Numbers
File: CIK_NAME_TICKER_EXCHANGE.csv
Source: U.S. Securities and Exchange Commission (SEC)

Documentation: https://www.sec.gov/edgar/searchedgar/accessing-edgar-data.htm

Description: Official registry of all publicly traded companies registered with the SEC, including unique identifiers for regulatory filings.
Coverage:

All SEC-registered public companies
Total records: ~13,000+ companies

Update Frequency: Updated by SEC as companies register/delist
Last Downloaded: [Date of your download]
Key Fields:

CIK: Central Index Key (unique SEC identifier, 10-digit number)
NAME: Official company name as registered with SEC
TICKER: Stock ticker symbol
EXCHANGE: Trading exchange (e.g., NYSE, NASDAQ, OTC)

Purpose in Project:

Link breach data to publicly traded companies
Enable retrieval of SEC filings (10-K, 8-K, proxy statements)
Support stock market impact analysis
Provide official company names and identifiers
Enable industry classification via SIC codes (if included in SEC data)

Data Quality Notes:

Contains only public companies (private companies, nonprofits, government entities will not match)
Some breached organizations are subsidiaries - may not have independent tickers
Company name variations may complicate matching
Delisted companies remain in historical data but may not be current

Coverage:

Subset of breached organizations that are publicly traded
Expected match rate: ~15-25% of total breached organizations (most are private companies, small businesses, or nonprofits)

Key Fields:

normalized_org_name: Standardized company name from breach data
ticker: Stock ticker symbol matched to SEC data
Additional fields as created: match_confidence, notes, verification_date, etc.

Purpose in Project:

Bridge between breach data and SEC financial/regulatory data
Enable company-specific financial analysis for public firms
Support event studies on market reactions to breach disclosures
Link to stock price data, financial statements, and SEC filings

Maintenance:

Updated as new publicly traded companies experience breaches
Requires periodic verification against current SEC listings
Manual review needed for mergers, acquisitions, name changes, delistings

Limitations:

Manual matching is time-intensive and may contain errors
Subsidiaries of public companies may be challenging to match
DBA (doing business as) names complicate matching
Private equity acquisitions remove companies from public markets


Data Integration Flow
Data_Breach_Chronology.xlsx (74,797 raw records)
         ↓
    dataclean.py
         ├─ Remove UNKN breach types
         ├─ Select 20 core columns
         ├─ Convert dates to datetime
         └─ Convert numeric fields
         ↓
    Cleaned dataset (~35,378 records)
         ↓
    dataload.py
         ├─ Save to CSV (output/databreach.csv)
         └─ Load to SQLite (databreach.db → databreach table)
         ↓
    Merge with CIK_NAME_TICKER_EXCHANGE.csv
         └─ Join on ticker
         ↓
    databreach_enriched table
         └─ Includes CIK, exchange, official company names

Data Limitations
Breach Chronology Limitations

Reporting bias: Only includes publicly disclosed breaches reported to government agencies
Voluntary reporting: Breaches below state thresholds may not be reported
Geographic coverage: U.S. only; no international breaches unless affecting U.S. residents
Time lag: Delay between breach occurrence, discovery, and public disclosure (median ~30-90 days)
Incomplete information: Some fields contain missing or unknown values
Classification subjectivity: AI and human classification may introduce errors
Not comprehensive: Does not represent all breaches, only those publicly reported to these 15 agencies
Grouping imperfect: Related breaches may not be grouped; unrelated breaches may be grouped

SEC Data Limitations

Public companies only: No coverage of private companies, nonprofits, government entities, or small businesses
Name matching challenges: Subsidiaries, DBA names, mergers, and acquisitions complicate matching
Delisted companies: Historical breaches may involve companies no longer trading
International entities: Limited coverage of non-U.S. companies (only those with SEC registration)
Timing: CIK assignments change with corporate events (mergers, spin-offs, bankruptcy)

Ticker Matching Limitations

Manual effort: Labor-intensive, not fully automated
Ambiguous names: Generic organization names difficult to match uniquely
Coverage gaps: Estimated 75-85% of breached organizations are private and cannot be matched
Verification needed: Matches should be spot-checked for accuracy
Temporal issues: Companies may have changed tickers or been delisted since breach


Citation and Attribution
For Academic Use
When using this data in research, proper citation is required:
Breach Data Citation:
Privacy Rights Clearinghouse. (2025). Data Breach Chronology 
(Version 2.1) [Database]. Retrieved from https://privacyrights.org
SEC Data Citation:
U.S. Securities and Exchange Commission. (2025). Company Tickers. 
Licensing

Data Breach Chronology: Subject to Privacy Rights Clearinghouse Terms of Service
SEC Data: Public domain (U.S. government data)


Data Access and Availability
Public Availability

Breach Chronology: Publicly available from Privacy Rights Clearinghouse
SEC Data: Publicly available (free, no registration required)
Company Ticker Matches: Project-specific (created for this dissertation research)

Reproducibility
To reproduce this dataset:

Download Data Breach Chronology from Privacy Rights Clearinghouse
Download SEC company tickers JSON from SEC website
Run dataclean.py to clean breach data
Run dataload.py to create SQLite database
Create ticker matching file (manual or semi-automated process)
Merge datasets using enrichment scripts


Data Updates and Maintenance
Refresh Schedule

Breach data: One-time snapshot (June 2025) for dissertation research
SEC data: One-time download; could be refreshed quarterly if needed
Ticker matches: Updated as new public companies are breached during research period

Version Control

Dataset version: Data Breach Chronology v2.1 (May 2025)
Analysis snapshot: June 2025
Cleaned data version: As processed by dataclean.py


Contact and Questions
For questions about the breach data:

Corrections: databreachcorrections@privacyrights.org
Licensing: databreachchronology@privacyrights.org
General: https://privacyrights.org

For SEC data questions:

https://www.sec.gov/os/accessing-edgar-data