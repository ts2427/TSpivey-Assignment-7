# Analysis Methodology Documentation
## Data Breach Exploratory Data Analysis

**Author:** T. Spivey  
**Course:** BUS 761 - Business Analytics  
**Assignment:** 5 - Exploratory Data Analysis Module  
**Date:** October 21, 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Statistical Methods](#statistical-methods)
3. [Data Preparation](#data-preparation)
4. [Analytical Techniques](#analytical-techniques)
5. [Visualization Approach](#visualization-approach)
6. [Assumptions and Limitations](#assumptions-and-limitations)
7. [Quality Assurance](#quality-assurance)

---

## Overview

This document describes the comprehensive statistical methodology used to analyze 35,378 data breach incidents. Our approach combines descriptive statistics, inferential hypothesis testing, regression modeling, and time series analysis to uncover patterns and inform business decision-making.

### Research Questions

1. **Is there a relationship between total individuals affected and state residents affected?**
   - Methods: Pearson and Spearman correlation
   
2. **Do different organization types experience different breach types?**
   - Method: Chi-squared test of independence
   
3. **Does breach impact vary significantly across industries?**
   - Method: One-way ANOVA with post-hoc testing
   
4. **Can we predict resident impact from total impact?**
   - Methods: Simple and multiple linear regression
   
5. **What temporal patterns exist in breach frequency and severity?**
   - Method: Time series analysis

### Significance Level

All hypothesis tests use **α = 0.05** (95% confidence level) unless otherwise specified. This is standard practice in business analytics and provides a reasonable balance between Type I and Type II errors.

---

## Statistical Methods

### 1. Descriptive Statistics

**Purpose:** Summarize central tendency, dispersion, and distribution shape.

**Metrics Calculated:**
- **Central Tendency:** Mean, median, mode
- **Dispersion:** Standard deviation, variance, interquartile range (IQR)
- **Distribution Shape:** Skewness, kurtosis
- **Range:** Min, max, Q1, Q3
- **Relative Dispersion:** Coefficient of variation (CV)

**Formula Reference:**
```
Mean (μ) = Σx / n
Std Dev (σ) = √[Σ(x - μ)² / (n-1)]
Skewness = [n / (n-1)(n-2)] × Σ[(x - μ)/σ]³
Kurtosis = [n(n+1) / (n-1)(n-2)(n-3)] × Σ[(x - μ)/σ]⁴ - [3(n-1)² / (n-2)(n-3)]
IQR = Q3 - Q1
CV = (σ / μ) × 100%
```

**Interpretation Guidelines:**
- **Skewness:**
  - Near 0: Symmetric distribution
  - > 0: Right-skewed (long tail to right)
  - < 0: Left-skewed (long tail to left)
  
- **Kurtosis:**
  - Near 0: Normal distribution
  - > 0: Heavy tails (leptokurtic)
  - < 0: Light tails (platykurtic)

**Business Application:**  
Descriptive statistics provide baseline understanding of typical breach characteristics, helping organizations set realistic expectations and benchmarks.

---

### 2. Correlation Analysis

**Purpose:** Measure strength and direction of linear and monotonic relationships between continuous variables.

#### Pearson Correlation Coefficient (r)

**Assumptions:**
1. Variables are continuous
2. Linear relationship exists
3. Bivariate normality (for significance testing)
4. No extreme outliers

**Formula:**
```
r = Σ[(x - x̄)(y - ȳ)] / √[Σ(x - x̄)² × Σ(y - ȳ)²]
```

**Interpretation:**
- r = +1: Perfect positive linear relationship
- r = 0: No linear relationship
- r = -1: Perfect negative linear relationship

**Strength Guidelines:**
- |r| < 0.3: Weak
- 0.3 ≤ |r| < 0.7: Moderate
- |r| ≥ 0.7: Strong

**Significance Testing:**  
H₀: ρ = 0 (no correlation in population)  
H₁: ρ ≠ 0 (correlation exists in population)

Test statistic: t = r√(n-2) / √(1-r²), follows t-distribution with n-2 df

#### Spearman Rank Correlation (ρ)

**Advantages over Pearson:**
1. Does not assume linearity
2. Robust to outliers
3. Works with ordinal data
4. Detects monotonic relationships

**When to Use Spearman:**
- Non-linear but monotonic relationship
- Presence of extreme outliers
- Ordinal variables
- Non-normal distributions

**Formula:**
```
ρ = 1 - [6Σd²] / [n(n²-1)]
where d = difference between ranks
```

**Our Analysis:**  
We calculated both Pearson and Spearman correlations. The significant difference (r=0.32 vs ρ=0.52) indicates:
- Strong monotonic relationship
- Non-linear pattern
- Presence of influential outliers
- Breach data follows power-law distribution (common in security incidents)

---

### 3. Chi-Squared Test of Independence

**Purpose:** Determine if two categorical variables are independent or associated.

**Hypotheses:**
- H₀: Variables are independent (no relationship)
- H₁: Variables are associated (relationship exists)

**Test Statistic:**
```
χ² = Σ[(O - E)² / E]
where:
O = Observed frequency
E = Expected frequency under independence
```

**Expected Frequency Calculation:**
```
E_ij = (Row_i Total × Column_j Total) / Grand Total
```

**Degrees of Freedom:**
```
df = (r - 1)(c - 1)
where r = number of rows, c = number of columns
```

**Assumptions:**
1. ✅ Data are frequencies (counts)
2. ✅ Observations are independent
3. ✅ Sample size is adequate
4. ✅ Expected frequencies ≥ 5 in all cells

**Validation in Our Analysis:**
- Smallest expected frequency: 8.7 (exceeds minimum of 5)
- All cells meet assumption requirements
- Sample size: 35,378 (well above minimum)

**Interpretation:**
- χ² = 5,069.93, df = 42, p < 0.001
- **Conclusion:** Strong evidence that organization type and breach type are related

**Effect Size (Cramér's V):**
```
V = √[χ² / (n × min(r-1, c-1))]
V = √[5069.93 / (35378 × 6)] = 0.15 (small to medium effect)
```

**Business Interpretation:**  
While statistically significant, the practical importance comes from examining specific cell deviations (observed vs expected) to identify industry-specific vulnerabilities.

---

### 4. One-Way ANOVA

**Purpose:** Test if means differ across three or more groups.

**Hypotheses:**
- H₀: μ₁ = μ₂ = ... = μₖ (all group means are equal)
- H₁: At least one group mean differs

**Test Statistic:**
```
F = MS_between / MS_within
where:
MS_between = SS_between / df_between
MS_within = SS_within / df_within

SS_between = Σnᵢ(x̄ᵢ - x̄_grand)²
SS_within = ΣΣ(xᵢⱼ - x̄ᵢ)²
```

**Assumptions:**
1. Independence of observations ✅
2. Normality of residuals (somewhat robust to violations)
3. Homogeneity of variances (Levene's test can check)

**Our Results:**
- F = 2.65, df = (6, 11548), p = 0.010
- **Conclusion:** Breach impact varies significantly across organization types

**Post-Hoc Testing:**  
When ANOVA is significant, Tukey HSD test identifies which specific groups differ:

```
HSD = q × √(MS_within / n)
where q = studentized range statistic
```

**Limitations:**
- Assumes equal variances (Welch's ANOVA available if violated)
- Sensitive to outliers (robust alternatives: Kruskal-Wallis H test)
- Multiple comparisons increase Type I error (corrected by Tukey adjustment)

---

### 5. Linear Regression

**Purpose:** Model relationship between predictor(s) and continuous outcome.

#### Simple Linear Regression

**Model:**
```
y = β₀ + β₁x + ε
where:
y = dependent variable (residents_affected)
x = independent variable (total_affected)
β₀ = intercept
β₁ = slope
ε = error term
```

**Assumptions:**
1. **Linearity:** Relationship is linear
2. **Independence:** Observations are independent
3. **Homoscedasticity:** Constant error variance
4. **Normality:** Errors are normally distributed

**Coefficient Estimation (Ordinary Least Squares):**
```
β₁ = Σ[(x - x̄)(y - ȳ)] / Σ(x - x̄)²
β₀ = ȳ - β₁x̄
```

**Model Evaluation:**

**R² (Coefficient of Determination):**
```
R² = 1 - (SS_residual / SS_total)
R² = 0.099 (9.9% of variance explained)
```

**Interpretation:**
- R² near 0: Poor model fit
- R² near 1: Excellent model fit
- Our R² = 0.099: Simple model insufficient; need additional predictors

**Residual Analysis:**
- Plot residuals vs fitted values (check homoscedasticity)
- QQ plot of residuals (check normality)
- Residuals vs leverage (identify influential points)

#### Multiple Linear Regression

**Model:**
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₖxₖ + ε
```

**Advantages:**
- Controls for confounding variables
- Higher R² (but beware overfitting)
- Tests individual variable contributions

**Challenges with Categorical Predictors:**
- One-hot encoding required
- Reference category interpretation
- Multicollinearity concerns

---

### 6. Logistic Regression

**Purpose:** Model binary outcomes (severe vs non-severe breaches).

**Model:**
```
log(p / (1-p)) = β₀ + β₁x₁ + ... + βₖxₖ
where:
p = P(severe breach)
log(p/(1-p)) = log-odds (logit)
```

**Interpretation:**
- **Odds Ratio:** exp(β)
  - OR > 1: Increases odds of severe breach
  - OR < 1: Decreases odds
  - OR = 1: No effect

**Model Evaluation:**
- **Accuracy:** (TP + TN) / Total
- **Precision:** TP / (TP + FP)
- **Recall:** TP / (TP + FN)
- **AUC-ROC:** Area under receiver operating characteristic curve

**Our Results:**
- Accuracy: 63%
- Threshold: 10,000 individuals affected
- Interpretation: Model performs moderately better than random chance (50%)

---

### 7. Time Series Analysis

**Purpose:** Identify temporal patterns and trends.

**Components:**
1. **Trend:** Long-term increase or decrease
2. **Seasonality:** Regular periodic fluctuations
3. **Cyclical:** Non-periodic fluctuations
4. **Random:** Irregular variations

**Methods Used:**
- **Aggregation:** Group by year/month
- **Moving Averages:** Smooth short-term fluctuations
- **Trend Testing:** Linear regression on time
- **Change Point Detection:** Identify structural breaks

**Visualization:**
- Line plots for trends
- Bar charts for comparisons
- Log scale for exponential growth

---

## Data Preparation

### Data Cleaning Process

**Steps Performed:**
1. **Column Selection:** Reduced from 37 to 20 essential columns
2. **Filtering:** Removed UNKN breach types (1,419 records)
3. **Type Conversion:**
   - Dates: String → datetime64
   - Numeric: String → float64
4. **Missing Value Handling:** Coercion (invalid → NaT/NaN)

**Cleaning Rationale:**
- **Coercion vs Deletion:** Preserved records with partial information
- **UNKN Removal:** Cannot analyze unknown breach types
- **No Imputation:** Conservative approach; prefer missing to incorrect data

### Missing Data Analysis

**Patterns Identified:**
- **MCAR** (Missing Completely At Random): Date fields
- **MAR** (Missing At Random): Impact metrics
- **MNAR** (Missing Not At Random): Location details

**Handling Strategy:**
- **Listwise deletion:** For correlation/regression (requires complete cases)
- **Available case analysis:** For descriptive statistics by group
- **No imputation:** Missing data reflects true reporting limitations

---

## Analytical Techniques

### Univariate Analysis

**Continuous Variables:**
- Histograms (frequency distributions)
- Box plots (identify outliers)
- Density plots (distribution shape)
- Summary statistics

**Categorical Variables:**
- Frequency tables
- Bar charts
- Pie charts (only for <7 categories)

### Bivariate Analysis

**Continuous × Continuous:**
- Scatter plots
- Correlation coefficients
- Regression lines

**Categorical × Categorical:**
- Contingency tables
- Stacked bar charts
- Heatmaps
- Chi-squared tests

**Categorical × Continuous:**
- Box plots by group
- Violin plots
- Group means/medians
- ANOVA / t-tests

### Multivariate Analysis

**Techniques:**
- Multiple regression
- Logistic regression
- Polynomial regression
- Regularized regression (Ridge, Lasso)

---

## Visualization Approach

### Design Principles

1. **Clarity:** Simple, uncluttered designs
2. **Accuracy:** Honest representation of data
3. **Efficiency:** Quick comprehension
4. **Aesthetics:** Professional appearance

### Chart Selection Guidelines

| Data Type | Best Chart | Use Case |
|-----------|-----------|----------|
| Single continuous | Histogram, Box plot | Distribution |
| Single categorical | Bar chart | Frequencies |
| Time series | Line chart | Trends |
| Two continuous | Scatter plot | Relationship |
| Categorical × Continuous | Box plot, Violin | Group comparison |
| Categorical × Categorical | Heatmap, Grouped bar | Cross-tabulation |

### Color Schemes

- **Sequential:** Single hue for ordered data (light → dark)
- **Diverging:** Two hues for data with midpoint (blue ← white → red)
- **Qualitative:** Distinct hues for categories (avoid red-green for accessibility)

### Business-Focused Formatting

- **Titles:** Clear, action-oriented
- **Labels:** Units specified
- **Annotations:** Highlight key findings
- **Legends:** Positioned for easy reference
- **Resolution:** 300 DPI for publication

---

## Assumptions and Limitations

### Statistical Assumptions

**Addressed:**
✅ Independence of observations (breach events are independent)  
✅ Adequate sample size (n = 35,378)  
✅ Appropriate measurement levels (continuous, categorical)

**Violated or Uncertain:**
⚠️ Normality (distributions are right-skewed; used robust methods)  
⚠️ Homoscedasticity (variance not constant; used log transformations)  
⚠️ Linearity (non-linear patterns present; addressed with Spearman correlation)

### Data Limitations

1. **Reporting Bias:** Only publicly disclosed breaches
2. **Time Lag:** 30-90 day median delay in reporting
3. **Geographic Scope:** U.S. only
4. **Classification Errors:** AI categorization ~5-10% error rate
5. **Missing Data:** 40% incomplete impact metrics

### Analytical Limitations

1. **Correlation ≠ Causation:** Associations identified, not causal mechanisms
2. **External Validity:** Findings may not generalize beyond U.S. breach landscape
3. **Temporal Validity:** Historical patterns may not predict future threats
4. **Model Simplicity:** Simple models used; more complex techniques available

---

## Quality Assurance

### Validation Steps

1. **Data Validation:**
   - Cross-checked sample records against source
   - Verified statistical calculations manually
   - Confirmed database schema matches data dictionary

2. **Code Review:**
   - Modular functions with docstrings
   - Unit tests for key calculations
   - Peer review of analytical approach

3. **Results Validation:**
   - Compared with published breach statistics
   - Sanity checks on all metrics
   - Verified visualizations match underlying data

### Reproducibility

**All analyses are fully reproducible:**
- ✅ Version-controlled code (Git)
- ✅ Documented data sources
- ✅ Random seeds set (where applicable)
- ✅ Package versions specified
- ✅ Clear step-by-step methodology

### Ethical Considerations

- **Privacy:** No PII included in analysis
- **Transparency:** Methods fully documented
- **Honesty:** Limitations clearly stated
- **Objectivity:** Findings reported without bias

---

## References

### Statistical Methods
- Field, A. (2018). *Discovering Statistics Using IBM SPSS Statistics* (5th ed.)
- Agresti, A. (2018). *Statistical Methods for the Social Sciences* (5th ed.)
- James, G., et al. (2021). *An Introduction to Statistical Learning* (2nd ed.)

### Data Breach Research
- Ponemon Institute. (2024). *Cost of a Data Breach Report*
- Verizon. (2024). *Data Breach Investigations Report*
- Privacy Rights Clearinghouse. (2025). *Data Breach Chronology*

### Python Libraries
- pandas: McKinney, W. (2022). Python for Data Analysis (3rd ed.)
- scipy: Virtanen, P., et al. (2020). SciPy 1.0 Nature Methods
- seaborn: Waskom, M. (2021). seaborn: statistical data visualization

---

## Appendix: Statistical Formulas Summary

```python
# Descriptive Statistics
mean = sum(x) / len(x)
variance = sum((x - mean)**2) / (len(x) - 1)
std_dev = sqrt(variance)
skewness = moment_3 / std_dev**3
kurtosis = moment_4 / std_dev**4 - 3

# Pearson Correlation
r = cov(x, y) / (std(x) * std(y))

# Spearman Correlation
rho = 1 - (6 * sum(d**2)) / (n * (n**2 - 1))

# Chi-Squared
chi2 = sum((observed - expected)**2 / expected)

# ANOVA F-statistic
F = MS_between / MS_within

# Linear Regression
beta_1 = cov(x, y) / var(x)
beta_0 = mean(y) - beta_1 * mean(x)
R_squared = 1 - (SS_residual / SS_total)
```

---

**Document Version:** 1.0  
**Last Updated:** October 21, 2025  
**Maintained by:** T. Spivey, BUS 761
