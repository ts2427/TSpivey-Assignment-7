# Data Breach Analytics Dashboard - User Guide

**Version:** 1.0  
**Last Updated:** October 24, 2025  
**Author:** T. Spivey

---

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Navigation](#navigation)
4. [Using Each Page](#using-each-page)
5. [Interpreting Results](#interpreting-results)
6. [Frequently Asked Questions](#frequently-asked-questions)
7. [Troubleshooting](#troubleshooting)

---

## Introduction

The Data Breach Analytics Dashboard is a comprehensive tool for analyzing data breach incidents, predicting breach severity, and generating business risk assessments. This dashboard integrates:

- **35,378 breach incidents** from 2003-2025
- **Statistical analysis** with correlation and hypothesis testing
- **Machine learning models** with 87% accuracy
- **Business recommendations** tailored to your industry

### Who Should Use This Dashboard?

- **Security Teams** - Monitor breach trends and assess vulnerabilities
- **Risk Managers** - Quantify financial impact and prioritize resources
- **Executives** - Make informed decisions about security investments
- **Analysts** - Explore data patterns and generate reports

---

## Getting Started

### Accessing the Dashboard

**Method 1: Local Installation**
```bash
cd DataBreach
streamlit run app.py
```
The dashboard will open in your browser at `http://localhost:8501`

**Method 2: Docker Container**
```bash
docker-compose up
```
Access at `http://localhost:8501`

### System Requirements

- **Browser:** Chrome, Firefox, Safari, or Edge (latest version)
- **Internet:** Required for initial load only
- **Screen Resolution:** 1280x720 minimum (1920x1080 recommended)

---

## Navigation

### Sidebar Menu

The dashboard has 5 main sections accessible from the left sidebar:

1. **🏠 Home** - Overview and quick stats
2. **📊 Data Explorer** - Interactive data exploration with filters
3. **📈 Statistical Analysis** - Key findings from Assignment 5
4. **🤖 ML Predictions** - Predict breach severity
5. **💼 Risk Assessment** - Generate business recommendations

### Tips for Navigation

- Use the sidebar to switch between pages
- Filters apply only to the current page
- Your selections are remembered within a session
- Refresh the page (F5) to reset all filters

---

## Using Each Page

### 1. Home Page

**Purpose:** Get an overview of the system and recent breach activity.

**Key Features:**
- **Quick Stats** - Total breaches, date range, organization types
- **Project Overview** - How all assignments integrate together
- **Recent Breaches** - Latest 10 breach incidents

**What to Do:**
1. Review the quick stats to understand the dataset
2. Check recent breaches for current trends
3. Use the sidebar to navigate to other sections

---

### 2. Data Explorer

**Purpose:** Explore breach data with interactive filters and visualizations.

#### Using Filters

**In the Sidebar:**
- **Year Range** - Drag slider to select time period
- **Organization Type** - Filter by industry (MED, BSF, BSR, etc.)
- **Breach Type** - Filter by attack method (HACK, DISC, PHYS, etc.)

**Filter Tips:**
- Start broad, then narrow down
- Combine filters for specific insights
- Watch the metrics update in real-time

#### Summary Metrics

Four key metrics update based on your filters:
- **Filtered Breaches** - Number of incidents matching filters
- **Avg Impact** - Average individuals affected
- **Median Impact** - Middle value (less affected by outliers)
- **Max Impact** - Largest breach in filtered set

#### Visualization Tabs

**Time Series Tab:**
- Shows breach frequency over time
- Useful for identifying trends and peaks
- Monthly view shows last 3 years of data

**Organization Types Tab:**
- Bar chart shows distribution by industry
- Pie chart shows proportions
- Identify which industries are most affected

**Breach Types Tab:**
- Bar chart shows distribution by attack method
- Treemap provides hierarchical view
- See which threats are most common

**Geographic Tab:**
- Shows top 20 states by breach count
- Identifies geographic hotspots
- Useful for regional security planning

#### Data Table

- Scrollable table shows filtered results
- Click column headers to sort
- Use **Download** button to export CSV

**Download Tips:**
- Exports currently filtered data only
- File named with current date
- Opens in Excel or any CSV reader

---

### 3. Statistical Analysis

**Purpose:** View key statistical findings from exploratory data analysis.

#### Correlation Analysis

**What It Shows:**
- Relationship between total affected and residents affected
- Pearson (linear) vs Spearman (monotonic) correlation

**How to Interpret:**
- **Pearson r=0.315** - Moderate linear relationship
- **Spearman ρ=0.517** - Stronger rank-order relationship
- **Key Finding:** Non-linear pattern with outliers

**Why It Matters:**
- Helps predict state-level impact from total numbers
- Identifies when breaches disproportionately affect certain states

#### Chi-Squared Test

**What It Shows:**
- Whether organization type and breach type are related
- Statistical significance testing

**How to Interpret:**
- **χ²=5,069.93** - Very large test statistic
- **p<0.001** - Highly statistically significant
- **Result:** Strong evidence that different industries face different threats

**Why It Matters:**
- Justifies industry-specific security strategies
- Validates tailored recommendations

**Using the Heatmap:**
- Darker colors = more breaches
- Look for patterns across rows/columns
- Identify your industry's vulnerabilities

#### ANOVA Results

**What It Shows:**
- Whether breach impact differs by organization type
- Comparison across multiple groups

**How to Interpret:**
- **F=2.65, p=0.010** - Statistically significant
- **Box Plot** - Shows distribution for each industry
- **Key Finding:** Healthcare and financial have different risk profiles

**Why It Matters:**
- Helps set realistic expectations by industry
- Informs budget allocation decisions

#### Descriptive Statistics Table

Shows summary stats by organization type:
- **Count** - Number of breaches
- **Mean** - Average impact
- **Median** - Typical impact
- **Std Dev** - Variability
- **Min/Max** - Range

---

### 4. ML Predictions

**Purpose:** Predict whether a potential breach will be severe (>10,000 affected).

#### Model Performance

Review the metrics to understand model reliability:
- **Accuracy: 87.0%** - Correct 87% of the time
- **Precision: 63.3%** - Of predicted severe, 63% are actually severe
- **Recall: 83.0%** - Catches 83% of all severe breaches
- **F1 Score: 0.718** - Balanced performance measure

**What This Means:**
- Model is reliable for most predictions
- Some false alarms (37% of severe predictions)
- Rarely misses actual severe breaches (17%)

#### Making a Prediction

**Step 1: Select Organization Type**
Choose from dropdown:
- MED (Healthcare)
- BSF (Financial Services)
- BSR (Retail)
- GOV (Government)
- EDU (Education)
- BSO (Business/Other)
- NGO (Nonprofit)

**Step 2: Select Breach Type**
Choose attack method:
- HACK (Hacking/Cyber Attack)
- DISC (Disclosure/Misconfiguration)
- PHYS (Physical Theft/Loss)
- CARD (Payment Card Compromise)
- INSD (Insider Threat)
- PORT (Portable Device)
- STAT (Stationary Device)

**Step 3: Select Breach Date**
Use date picker to choose when breach occurred/will occur.

**Step 4: Click "Predict Severity"**

#### Understanding Prediction Results

**Severity Probability:**
- 0-30%: Low risk
- 30-70%: Medium risk
- 70-100%: High risk

**Classification:**
- **SEVERE** - Likely to affect >10,000 people
- **Non-Severe** - Likely to affect <10,000 people

**Risk Level:**
- **High** - Immediate action required
- **Medium** - Enhanced monitoring
- **Low** - Routine procedures

**Gauge Chart:**
- Visual representation of probability
- Green zone: Low risk
- Yellow zone: Medium risk
- Red zone: High risk

#### Feature Importance

Shows which factors most influence predictions:
- **breach_type_HACK** - Hacking attacks (most important)
- **organization_type_MED** - Healthcare industry
- **breach_year** - Temporal trends
- etc.

**Why This Matters:**
- Understand what drives severe breaches
- Focus mitigation efforts on high-importance factors

---

### 5. Risk Assessment

**Purpose:** Generate comprehensive business risk assessments and recommendations.

#### Setting Up a Scenario

**Step 1: Organization Type**
Select your industry from dropdown.

**Step 2: Severity Probability**
Use slider to set risk level (0.0 - 1.0):
- Use ML Predictions page result, or
- Enter your own estimate

**Step 3: Predicted Impact**
Enter number of individuals likely affected:
- Minimum: 100
- Maximum: 1,000,000
- Default: 25,000

**Step 4: Click "Generate Risk Assessment"**

#### Understanding the Results

**Risk Level Banner:**
- 🔴 **Critical** (70-100%) - Emergency response needed
- 🟠 **High** (50-70%) - Immediate action required
- 🟡 **Medium** (30-50%) - Enhanced monitoring
- 🟢 **Low** (0-30%) - Routine procedures

**Key Metrics:**
1. **Severity Probability** - Your input
2. **Predicted Impact** - Number of individuals
3. **Estimated Cost** - Total financial impact
4. **Priority Score** - 0-100 ranking

#### Cost Breakdown

**Four Cost Categories:**

1. **Direct Costs (Records)**
   - $225 per affected individual
   - Industry average from Ponemon Institute
   - Includes investigation, remediation, victim support

2. **Notification Costs**
   - $5 per individual
   - Letter printing, postage, call center
   - Regulatory requirement

3. **Legal/Regulatory**
   - $50 per individual
   - Potential fines, legal fees, settlements
   - GDPR/CCPA compliance costs

4. **PR/Recovery**
   - $25 per individual
   - Reputation management, customer retention
   - Marketing campaigns, credit monitoring

**Pie Chart:**
Visual representation of cost distribution.

#### Recommendations

Industry-specific action items, such as:

**For Healthcare (MED):**
- 🔴 CRITICAL items require immediate action
- Standard items are best practices
- 💰 Budget recommendations (% of IT spend)

**For Financial (BSF):**
- Focus on physical security
- Document handling procedures
- Digital transformation initiatives

**For Retail (BSR):**
- POS system hardening
- Payment security upgrades
- Fraud detection systems

#### Security Budget Calculator

**Example Calculation Tool:**
1. Enter your annual IT budget
2. System calculates recommended security spend
3. Based on industry-specific risk profile

**Budget Ranges by Industry:**
- Healthcare: 8-12% of IT budget
- Financial: 6-10%
- Retail: 6-8%
- Government: 7-10%
- Education: 5-7%
- Business/Other: 7-9%
- Nonprofit: 5-6%

#### Industry-Specific Insights Tabs

**Three Industry Deep Dives:**

1. **Healthcare Tab**
   - Primary threat: Disclosure breaches
   - 43% excess frequency
   - Key recommendations

2. **Financial Tab**
   - Primary threat: Physical breaches
   - 169% excess frequency
   - Security protocols

3. **Retail Tab**
   - Primary threat: Card breaches
   - 400% excess frequency
   - POS protection measures

---

## Interpreting Results

### Statistical Significance

**P-Value Interpretation:**
- p < 0.001: Highly significant (very strong evidence)
- p < 0.01: Significant (strong evidence)
- p < 0.05: Significant (sufficient evidence)
- p ≥ 0.05: Not significant (insufficient evidence)

**When p < 0.05:**
- Results unlikely due to chance
- Relationship/difference is real
- Can act on findings with confidence

### Confidence in Predictions

**High Confidence:**
- ML probability: 0-30% or 70-100%
- Clear classification
- Consistent with feature importance

**Medium Confidence:**
- ML probability: 30-70%
- Near decision boundary
- Consider additional factors

**When to Be Cautious:**
- Unusual combinations (rare org type + rare breach type)
- Very recent dates (less historical data)
- Extreme impact numbers (>500K individuals)

### Using Multiple Sources

**Best Practice:**
1. Check Data Explorer for historical patterns
2. Review Statistical Analysis for relationships
3. Run ML Prediction for specific scenario
4. Generate Risk Assessment for recommendations

**Triangulate findings across all sections.**

---

## Frequently Asked Questions

### General Questions

**Q: How often is the data updated?**
A: The current dataset includes breaches through June 2025. The database can be refreshed by re-running the ETL pipeline with updated source data.

**Q: Can I export data?**
A: Yes! Use the Download button on the Data Explorer page to export filtered data as CSV.

**Q: Does the dashboard require internet?**
A: No. Once loaded, the dashboard runs entirely offline (except initial page load).

**Q: Can I use this on mobile?**
A: The dashboard is optimized for desktop/laptop. Mobile viewing is possible but not recommended due to complex visualizations.

### Data Questions

**Q: What does "UNKN" breach type mean?**
A: Unknown breach types were filtered out during data cleaning (Assignment 4). The analysis only includes classified breaches.

**Q: Why are some "total_affected" values N/A?**
A: Not all breach notifications include impact numbers. This represents ~40% of records.

**Q: What's the difference between total_affected and residents_affected?**
A: total_affected = all individuals impacted nationwide/globally. residents_affected = only individuals in the reporting state.

**Q: Are these all U.S. breaches?**
A: Yes. Data source (Privacy Rights Clearinghouse) tracks breaches reported to U.S. government agencies.

### Prediction Questions

**Q: How accurate are the predictions?**
A: The Random Forest model achieves 87% overall accuracy with 83% recall for severe breaches.

**Q: Can I trust a "severe" prediction?**
A: 63% of severe predictions are correct (precision). Use as one input for decision-making, not the only factor.

**Q: What if my organization type isn't listed?**
A: Choose the closest match:
  - Tech company → BSO
  - Healthcare-adjacent → MED
  - Financial services → BSF

**Q: Can I predict breaches more than 1 year out?**
A: The model is trained on historical data through 2025. Predictions far into the future become less reliable as threat landscape evolves.

### Risk Assessment Questions

**Q: Are cost estimates accurate?**
A: Costs are based on 2024 industry averages (Ponemon Institute). Actual costs vary by organization size, industry, and breach specifics.

**Q: Should I follow all recommendations?**
A: Recommendations are prioritized (🔴 CRITICAL first). Implement based on your risk tolerance and budget.

**Q: How do I know what security budget is right?**
A: Use the recommended % range as a starting point, then adjust based on:
  - Your risk appetite
  - Regulatory requirements
  - Recent breach activity in your industry
  - Available resources

---

## Troubleshooting

### Dashboard Won't Load

**Problem:** Blank page or loading spinner forever

**Solutions:**
1. Check that `databreach.db` exists in the project folder
2. Verify Streamlit is running (check terminal for errors)
3. Try a different browser
4. Clear browser cache (Ctrl+F5)
5. Restart Streamlit: Ctrl+C, then `streamlit run app.py`

### Visualizations Not Appearing

**Problem:** Charts show as blank or error messages

**Solutions:**
1. Ensure Plotly is installed: `pip install plotly`
2. Check browser console for JavaScript errors (F12)
3. Try zooming out (Ctrl + minus key)
4. Disable browser extensions that block scripts

### Filters Not Working

**Problem:** Selecting filters doesn't change the data

**Solutions:**
1. Ensure you're on the Data Explorer page (not Statistical Analysis)
2. Check that filtered data count > 0
3. Refresh page (F5) to reset filters
4. Try broader filter combinations

### Predictions Fail

**Problem:** Error when clicking "Predict Severity"

**Solutions:**
1. Ensure all three inputs are selected (org type, breach type, date)
2. Check date is within reasonable range (2003-2030)
3. Restart Streamlit if model failed to load

### Slow Performance

**Problem:** Dashboard is laggy or unresponsive

**Solutions:**
1. Close other browser tabs
2. Reduce filter complexity (fewer simultaneous filters)
3. On Data Explorer, use narrower date ranges
4. Restart Docker container if using Docker

### Data Doesn't Match Documentation

**Problem:** Numbers don't align with Assignment 5 findings

**Solutions:**
1. Verify you're using the correct `databreach.db` file
2. Check that data cleaning removed UNKN breach types
3. Ensure database has 35,378 records (run `get_table_info()`)
4. Re-run ETL pipeline if database is corrupted

---

## Getting Help

### Resources

- **Technical Documentation:** See `DEPLOYMENT.md` for system architecture
- **Code Repository:** [GitHub link]
- **Data Source:** Privacy Rights Clearinghouse (https://privacyrights.org)

### Contact

- **Developer:** T. Spivey
- **Course:** BUS 761 - Business Analytics
- **Email:** [Your email]

### Reporting Issues

When reporting problems, please include:
1. What you were trying to do
2. What happened instead
3. Error messages (if any)
4. Browser and version
5. Screenshot (if applicable)

---

**Document Version:** 1.0  
**Last Updated:** October 24, 2025  
**Course:** BUS 761 - Assignment 7