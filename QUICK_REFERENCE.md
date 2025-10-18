# Quick Reference Guide
## EDA Package for Data Breach Analysis

**Author:** T. Spivey | **Course:** BUS 761 | **Date:** October 2025

---

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage
```python
from eda_package import BreachAnalyzer, BreachVisualizer, DataLoader

# 1. Load data
loader = DataLoader('databreach.db')
df = loader.load_breach_data()

# 2. Analyze
analyzer = BreachAnalyzer(df)
results = analyzer.correlation_analysis()

# 3. Visualize
viz = BreachVisualizer(df)
viz.plot_breach_frequency()
```

---

## 📚 Class Reference

### DataLoader

#### Initialize
```python
loader = DataLoader(db_name='databreach.db')
```

#### Common Methods
```python
# Load main dataset
df = loader.load_breach_data()

# Load statistical results
corr_df = loader.load_statistical_results('correlation_results')

# Get table information
info = loader.get_table_info()

# Execute custom query
df = loader.execute_query("SELECT * FROM databreach WHERE organization_type='MED'")

# Get summary
summary = loader.get_breach_summary()
```

---

### BreachAnalyzer

#### Initialize
```python
analyzer = BreachAnalyzer(df, alpha=0.05)  # alpha = significance level
```

#### Statistical Methods

##### Correlation Analysis
```python
results = analyzer.correlation_analysis(
    var1='total_affected',
    var2='residents_affected'
)
# Returns: {'pearson_r', 'pearson_p', 'spearman_rho', 'spearman_p', ...}
```

##### Chi-Squared Test
```python
results = analyzer.chi_squared_test(
    var1='organization_type',
    var2='breach_type'
)
# Returns: {'chi2_statistic', 'p_value', 'observed', 'expected', ...}
```

##### ANOVA
```python
results = analyzer.anova_test(
    group_var='organization_type',
    value_var='total_affected'
)
# Returns: {'f_statistic', 'p_value', 'group_statistics', ...}
```

##### Descriptive Statistics
```python
# Overall statistics
stats = analyzer.descriptive_statistics()

# By group
stats = analyzer.descriptive_statistics(group_by='organization_type')
```

##### Simple Linear Regression
```python
results = analyzer.simple_linear_regression(
    X_var='total_affected',
    y_var='residents_affected'
)
# Returns: {'slope', 'intercept', 'r_squared', ...}
```

##### Time Series Analysis
```python
# Yearly aggregation
ts = analyzer.time_series_analysis(date_col='breach_date', freq='Y')

# Monthly aggregation
ts = analyzer.time_series_analysis(date_col='breach_date', freq='M')
```

##### Logistic Regression
```python
results = analyzer.logistic_regression_severity(threshold=10000)
# Returns: {'accuracy', 'coefficients', 'model', ...}
```

##### Business Insights
```python
insights = analyzer.get_business_insights()
# Returns: Dict of business-focused interpretations
```

---

### BreachVisualizer

#### Initialize
```python
viz = BreachVisualizer(
    df,
    output_dir='output/visualizations',
    dpi=300
)
```

#### Visualization Methods

##### Industry Vulnerability Heatmap
```python
chi_observed = loader.load_statistical_results('chi_squared_observed')
fig = viz.plot_industry_vulnerability(chi_observed, save=True)
```

##### Breach Frequency
```python
fig = viz.plot_breach_frequency(save=True)
```

##### Correlation Scatter
```python
fig = viz.plot_correlation_scatter(
    var1='total_affected',
    var2='residents_affected',
    correlation_stats=corr_results,  # Optional
    save=True
)
```

##### Sector Impact
```python
desc_stats = analyzer.descriptive_statistics(group_by='organization_type')
fig = viz.plot_sector_impact(desc_stats, save=True)
```

##### Time Series Trends
```python
time_series = loader.load_statistical_results('time_series_yearly')
fig = viz.plot_time_series_trends(time_series, save=True)
```

##### Regression Fit
```python
fig = viz.plot_regression_fit(
    regression_results=reg_results,  # Optional
    save=True
)
```

##### Complete Dashboard
```python
# Generate all visualizations at once
viz.create_comprehensive_dashboard(
    chi_observed=chi_observed,
    desc_stats=desc_stats,
    time_series=time_series,
    correlation_stats=corr_results,
    regression_results=reg_results
)
```

---

## 💡 Common Workflows

### Workflow 1: Basic EDA
```python
# Load data
loader = DataLoader()
df = loader.load_breach_data()

# Summary statistics
analyzer = BreachAnalyzer(df)
desc_stats = analyzer.descriptive_statistics()
print(desc_stats)

# Quick visualization
viz = BreachVisualizer(df)
viz.plot_breach_frequency()
```

### Workflow 2: Hypothesis Testing
```python
# Load and analyze
loader = DataLoader()
df = loader.load_breach_data()
analyzer = BreachAnalyzer(df, alpha=0.05)

# Test 1: Correlation
corr = analyzer.correlation_analysis()
print(f"Pearson r: {corr['pearson_r']:.3f}, p: {corr['pearson_p']:.6f}")

# Test 2: Independence
chi = analyzer.chi_squared_test('organization_type', 'breach_type')
print(f"Chi-squared: {chi['chi2_statistic']:.2f}, p: {chi['p_value']:.6f}")

# Test 3: Group differences
anova = analyzer.anova_test('organization_type', 'total_affected')
print(f"F-statistic: {anova['f_statistic']:.2f}, p: {anova['p_value']:.6f}")
```

### Workflow 3: Complete Analysis & Reporting
```python
# 1. Load data
loader = DataLoader()
df = loader.load_breach_data()
chi_observed = loader.load_statistical_results('chi_squared_observed')
time_series = loader.load_statistical_results('time_series_yearly')

# 2. Perform all analyses
analyzer = BreachAnalyzer(df)
corr_results = analyzer.correlation_analysis()
chi_results = analyzer.chi_squared_test('organization_type', 'breach_type')
anova_results = analyzer.anova_test('organization_type', 'total_affected')
desc_stats = analyzer.descriptive_statistics(group_by='organization_type')
reg_results = analyzer.simple_linear_regression()

# 3. Generate visualizations
viz = BreachVisualizer(df)
viz.create_comprehensive_dashboard(
    chi_observed, desc_stats, time_series,
    corr_results, reg_results
)

# 4. Get business insights
insights = analyzer.get_business_insights()
for category, insight in insights.items():
    print(f"{category}: {insight}")
```

### Workflow 4: Custom Analysis
```python
# Load specific subset
loader = DataLoader()
query = """
SELECT * FROM databreach 
WHERE organization_type = 'MED' 
AND breach_date >= '2020-01-01'
"""
df_healthcare = loader.execute_query(query)

# Analyze subset
analyzer = BreachAnalyzer(df_healthcare)
stats = analyzer.descriptive_statistics()
print(f"Healthcare breaches 2020+: {len(df_healthcare)}")
print(f"Mean impact: {stats.loc[0, 'mean']:.0f}")
```

---

## 📊 Available Statistical Results Tables

Access pre-computed results from Assignment 4:

| Table Name | Description |
|-----------|-------------|
| `correlation_results` | Pearson & Spearman correlations |
| `chi_squared_summary` | Chi-squared test results |
| `chi_squared_observed` | Observed frequencies |
| `chi_squared_expected` | Expected frequencies |
| `anova_results` | ANOVA test results |
| `tukey_hsd_results` | Post-hoc pairwise comparisons |
| `descriptive_stats_by_org` | Descriptive stats by org type |
| `simple_regression_results` | Simple linear regression |
| `multiple_regression_results` | Multiple regression |
| `multiple_regression_coefficients` | Regression coefficients |
| `regularized_regression_results` | Ridge & Lasso |
| `polynomial_regression_results` | Polynomial models |
| `logistic_regression_results` | Logistic regression |
| `time_series_monthly` | Monthly aggregation |
| `time_series_yearly` | Yearly aggregation |

---

## 🎨 Customization Options

### Visualization Customization
```python
viz = BreachVisualizer(
    df,
    output_dir='custom/path',  # Custom output directory
    style='seaborn-v0_8-whitegrid',  # Different matplotlib style
    dpi=600  # Higher resolution
)

# Custom figure size
fig = viz.plot_breach_frequency(figsize=(16, 8))

# Don't save (return figure only)
fig = viz.plot_correlation_scatter(save=False)
plt.show()
```

### Analysis Customization
```python
# Different significance level
analyzer = BreachAnalyzer(df, alpha=0.01)  # 99% confidence

# Custom logistic regression threshold
results = analyzer.logistic_regression_severity(threshold=50000)

# Custom time series frequency
ts = analyzer.time_series_analysis(freq='M')  # Monthly instead of yearly
```

---

## 🐛 Troubleshooting

### Issue: "No module named 'eda_package'"
```python
# Solution: Add package to Python path
import sys
sys.path.append('/path/to/assignment5_eda')
from eda_package import BreachAnalyzer, BreachVisualizer, DataLoader
```

### Issue: "Database file not found"
```python
# Solution: Specify full path
loader = DataLoader('/full/path/to/databreach.db')
```

### Issue: "Figure not displaying"
```python
# Solution: Add plt.show() or save figure
import matplotlib.pyplot as plt
fig = viz.plot_breach_frequency(save=False)
plt.show()
```

### Issue: "Low memory warning"
```python
# Solution: Load specific columns only
query = "SELECT id, organization_type, breach_type FROM databreach"
df = loader.execute_query(query)
```

---

## 📖 Additional Resources

### Documentation
- **Executive Summary:** `documentation/executive_summary.md`
- **Methodology:** `documentation/analysis_methodology.md`
- **Full README:** `README_ASSIGNMENT5.md`

### Code Examples
- **Jupyter Notebook:** `notebooks/comprehensive_eda_analysis.ipynb`
- **Original Scripts:** `eda.py`, `visualizations.py`

### Get Help
- Check docstrings: `help(BreachAnalyzer.correlation_analysis)`
- Review notebook for examples
- Email: [Your Contact]

---

## 💻 Code Snippets Library

### Get Specific Industry Data
```python
loader = DataLoader()
df = loader.execute_query("""
    SELECT * FROM databreach 
    WHERE organization_type IN ('MED', 'BSF')
    AND total_affected > 1000
""")
```

### Compare Two Industries
```python
df_med = df[df['organization_type'] == 'MED']['total_affected']
df_bsf = df[df['organization_type'] == 'BSF']['total_affected']

from scipy import stats
t_stat, p_value = stats.ttest_ind(df_med.dropna(), df_bsf.dropna())
print(f"t-test: t={t_stat:.2f}, p={p_value:.4f}")
```

### Export Results to CSV
```python
# Get results
analyzer = BreachAnalyzer(df)
desc_stats = analyzer.descriptive_statistics(group_by='organization_type')

# Save to CSV
desc_stats.to_csv('descriptive_statistics.csv', index=True)
```

### Create Custom Visualization
```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='organization_type', y='total_affected')
plt.yscale('log')
plt.title('Breach Impact by Organization Type')
plt.savefig('custom_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()
```

---

**Version:** 1.0  
**Last Updated:** October 21, 2025  
**Maintained by:** T. Spivey, BUS 761

**Questions?** Check full documentation or contact for support!
