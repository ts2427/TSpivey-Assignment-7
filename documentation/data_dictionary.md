Data Dictionary - databreach Table
Table Overview

Table Name: databreach
Primary Key: id
Total Columns: 20
Total Rows: 35,378 (after cleaning)
Source: Privacy Rights Clearinghouse Data Breach Chronology v2.1


Column Specifications
Column NameData TypeDescriptionNullableConstraintsExample ValueidTEXTUnique identifier (UUID) for each breach recordNoPRIMARY KEY, UNIQUE"550e8400-e29b-41d4-a716-446655440000"org_nameTEXTOriginal organization name as reported in breach notificationYes-"Global Tech Solutions, Inc."normalized_org_nameTEXTStandardized organization name for tracking across multiple reportsYes-"Global Tech Solutions"reported_dateDATETIMEDate when breach was reported to government agencyYesFormat: YYYY-MM-DD"2024-03-15"breach_dateDATETIMEDate when breach occurred or was discoveredYesFormat: YYYY-MM-DD"2024-01-15"end_breach_dateDATETIMEDate when breach was contained/ended (if applicable)YesFormat: YYYY-MM-DD"2024-01-31"incident_detailsTEXTDetailed description of the breach event and circumstancesYes-"Unauthorized access to employee email account containing patient information"information_affectedTEXTTypes of personal information compromised in the breachYes-"Names, Social Security numbers, medical records"organization_typeTEXTClassification of the breached organizationYesValid values: BSF, BSO, BSR, EDU, GOV, MED, NGO"MED"breach_typeTEXTPrimary method or nature of the breachNoValid values: CARD, DISC, HACK, INSD, PHYS, PORT, STAT (UNKN excluded during cleaning)"HACK"group_org_breach_typeTEXTGrouped/consolidated breach type for related incidentsYesSame values as breach_type"HACK"group_org_typeTEXTGrouped/consolidated organization type for related incidentsYesSame values as organization_type"MED"total_affectedREALTotal number of individuals impacted across all jurisdictionsYesMust be >= 015000.0residents_affectedREALNumber of state residents affected (varies by reporting state)YesMust be >= 05000.0breach_location_streetTEXTStreet address where breach occurredYes-"5550 Peachtree Parkway, Suite 500"breach_location_cityTEXTCity where breach occurredYes-"Peachtree Corners"breach_location_stateTEXTState/province where breach occurredYes2-letter state code"GA"breach_location_zipTEXTZIP/postal code where breach occurredYes-"30092"breach_location_countryTEXTCountry where breach occurredYes-"United States"tagsTEXTAdditional categorization tags for the breachYesComma-separated values"sensitive-personal-information,finance"

Field Value Definitions
organization_type Values
CodeFull NameDescriptionBSFFinancial Services BusinessBanks, credit unions, investment firms, insurance carriers (excluding health)BSOOther BusinessTechnology companies, manufacturers, utilities, professional servicesBSRRetail BusinessPhysical and online retail merchantsEDUEducational InstitutionsSchools, universities, educational servicesGOVGovernment and MilitaryPublic administration, government agenciesMEDHealthcare ProvidersHospitals, clinics, HIPAA-covered entitiesNGONonprofitsCharities, advocacy groups, religious organizations
breach_type Values
CodeFull NameDescriptionCARDCard CompromisePhysical payment card compromises (skimming, POS tampering)DISCDisclosureUnintended disclosures (misconfiguration, accidents)HACKHacking/Cyber AttackExternal cyber attacks (malware, ransomware, network intrusions)INSDInsider ThreatInternal threats from authorized usersPHYSPhysical Theft/LossPhysical document theft or lossPORTPortable DevicePortable device breaches (laptops, phones, tablets)STATStationary DeviceStationary device breaches (desktops, servers)

Data Quality Notes
Missing Values

High missingness: breach_location_street (~70% missing)
Moderate missingness: end_breach_date (~40% missing)
Low missingness: reported_date, breach_date (<5% missing)
No missing values: id, breach_type (filtered during cleaning)

Data Transformations Applied

Date fields converted from text to datetime objects
Numeric fields (total_affected, residents_affected) converted from text to float
Invalid values coerced to NULL (NaT for dates, NaN for numbers)
Records with breach_type = 'UNKN' removed during cleaning

Data Validation Rules

All dates must be between 2005-01-01 and 2025-12-31
total_affected should be >= residents_affected (when both present)
breach_date should be <= reported_date (when both present)
organization_type and breach_type must match valid code lists


Usage Examples
Query breaches by type
sqlSELECT organization_type, COUNT(*) as breach_count
FROM databreach
GROUP BY organization_type
ORDER BY breach_count DESC;
Find large breaches
sqlSELECT org_name, breach_date, total_affected
FROM databreach
WHERE total_affected > 100000
ORDER BY total_affected DESC;
Filter by date range
sqlSELECT *
FROM databreach
WHERE breach_date BETWEEN '2023-01-01' AND '2023-12-31';