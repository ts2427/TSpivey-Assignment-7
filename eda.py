"""
Exploratory Data Analysis Module
Performs comprehensive statistical analysis on data breach patterns
"""

import pandas as pd
import sqlite3
import numpy as np
from scipy.stats import pearsonr, spearmanr, chi2_contingency, f_oneway, tukey_hsd
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

print("="*60)
print("EXPLORATORY DATA ANALYSIS")
print("="*60)

# Connect to the SQLite database
conn = sqlite3.connect('databreach.db', detect_types=sqlite3.PARSE_DECLTYPES)

# Load the data
databreach = pd.read_sql_query("SELECT * FROM databreach", conn)
print(f"Loaded {len(databreach):,} breach records")

# Convert numeric columns
databreach['total_affected'] = pd.to_numeric(databreach['total_affected'], errors='coerce')
databreach['residents_affected'] = pd.to_numeric(databreach['residents_affected'], errors='coerce')

# Convert date columns
databreach['breach_date'] = pd.to_datetime(databreach['breach_date'], errors='coerce')
databreach['reported_date'] = pd.to_datetime(databreach['reported_date'], errors='coerce')

# ===== CORRELATION ANALYSIS =====
print("\n" + "="*60)
print("CORRELATION ANALYSIS")
print("="*60)

valid_data = databreach[['total_affected', 'residents_affected']].dropna()
print(f"Valid pairs: {len(valid_data):,}")

# Calculate correlations
pearson_coef, pearson_pval = pearsonr(valid_data['total_affected'], 
                                       valid_data['residents_affected'])
spearman_coef, spearman_pval = spearmanr(valid_data['total_affected'], 
                                          valid_data['residents_affected'])

print(f"Pearson correlation: r={pearson_coef:.4f}, p={pearson_pval:.6f}")
print(f"Spearman correlation: rho={spearman_coef:.4f}, p={spearman_pval:.6f}")

# Save correlation results
correlation_results = pd.DataFrame({
    'test_type': ['Pearson', 'Spearman'],
    'variable_1': ['total_affected', 'total_affected'],
    'variable_2': ['residents_affected', 'residents_affected'],
    'coefficient': [pearson_coef, spearman_coef],
    'p_value': [pearson_pval, spearman_pval],
    'sample_size': [len(valid_data), len(valid_data)],
    'significant_at_0.05': [pearson_pval < 0.05, spearman_pval < 0.05]
})
correlation_results.to_sql('correlation_results', conn, if_exists='replace', index=False)
print("[OK] Correlation results saved")

# ===== CHI-SQUARED TEST =====
print("\n" + "="*60)
print("CHI-SQUARED TEST: Organization Type vs Breach Type")
print("="*60)

data_clean = databreach[['organization_type', 'breach_type']].dropna()
print(f"Valid observations: {len(data_clean):,}")

# Create contingency table
contingency_table = pd.crosstab(data_clean['organization_type'], 
                                 data_clean['breach_type'])

# Perform chi-squared test
chi2, p_value, dof, expected_freq = chi2_contingency(contingency_table)

print(f"Chi-squared statistic: {chi2:.4f}")
print(f"P-value: {p_value:.6f}")
print(f"Degrees of freedom: {dof}")
print(f"Result: {'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'} (alpha=0.05)")

# Save chi-squared results
chi_squared_summary = pd.DataFrame({
    'test_type': ['Chi-Squared'],
    'variable_1': ['organization_type'],
    'variable_2': ['breach_type'],
    'chi_squared_statistic': [chi2],
    'p_value': [p_value],
    'degrees_of_freedom': [dof],
    'sample_size': [len(data_clean)],
    'significant_at_0.05': [p_value < 0.05]
})
chi_squared_summary.to_sql('chi_squared_summary', conn, if_exists='replace', index=False)
print("[OK] Chi-squared summary saved")

# Save observed and expected frequencies
contingency_table.reset_index().to_sql('chi_squared_observed', conn, if_exists='replace', index=False)
print("[OK] Observed frequencies saved")

expected_df = pd.DataFrame(expected_freq, 
                          index=contingency_table.index, 
                          columns=contingency_table.columns).reset_index()
expected_df.to_sql('chi_squared_expected', conn, if_exists='replace', index=False)
print("[OK] Expected frequencies saved")

# ===== ANOVA =====
print("\n" + "="*60)
print("ANOVA: Breach Impact by Organization Type")
print("="*60)

# Prepare data for ANOVA
anova_data = databreach[['organization_type', 'total_affected']].dropna()
org_groups = [group['total_affected'].values for name, group in anova_data.groupby('organization_type')]

# Perform ANOVA
f_stat, anova_pval = f_oneway(*org_groups)

print(f"F-statistic: {f_stat:.4f}")
print(f"P-value: {anova_pval:.6f}")
print(f"Result: {'SIGNIFICANT' if anova_pval < 0.05 else 'NOT SIGNIFICANT'} (alpha=0.05)")

if anova_pval < 0.05:
    print("Interpretation: Breach impact differs significantly across organization types")

# Save ANOVA results
anova_summary = pd.DataFrame({
    'test_type': ['ANOVA'],
    'dependent_variable': ['total_affected'],
    'grouping_variable': ['organization_type'],
    'f_statistic': [f_stat],
    'p_value': [anova_pval],
    'num_groups': [len(org_groups)],
    'sample_size': [len(anova_data)],
    'significant_at_0.05': [anova_pval < 0.05]
})
anova_summary.to_sql('anova_results', conn, if_exists='replace', index=False)
print("[OK] ANOVA results saved")

# ===== POST-HOC TUKEY HSD =====
print("\n" + "="*60)
print("POST-HOC TUKEY HSD: Pairwise Comparisons")
print("="*60)

if anova_pval < 0.05:
    # Perform Tukey HSD
    tukey_result = tukey_hsd(*org_groups)
    
    # Get organization type names
    org_names = list(anova_data.groupby('organization_type').groups.keys())
    
    # Create pairwise comparison results
    tukey_comparisons = []
    for i in range(len(org_names)):
        for j in range(i+1, len(org_names)):
            p_val = tukey_result.pvalue[i, j]
            tukey_comparisons.append({
                'group_1': org_names[i],
                'group_2': org_names[j],
                'p_value': p_val,
                'significant': p_val < 0.05
            })
    
    tukey_df = pd.DataFrame(tukey_comparisons)
    tukey_df.to_sql('tukey_hsd_results', conn, if_exists='replace', index=False)
    
    print(f"Significant pairwise differences found:")
    sig_pairs = tukey_df[tukey_df['significant']]
    for _, row in sig_pairs.head(10).iterrows():
        print(f"  {row['group_1']} vs {row['group_2']}: p={row['p_value']:.4f}")
    print(f"[OK] Tukey HSD results saved ({len(sig_pairs)} significant pairs)")
else:
    print("Skipped (ANOVA not significant)")

# ===== DESCRIPTIVE STATISTICS =====
print("\n" + "="*60)
print("DESCRIPTIVE STATISTICS BY ORGANIZATION TYPE")
print("="*60)

desc_stats = databreach.groupby('organization_type')['total_affected'].agg([
    ('count', 'count'),
    ('mean', 'mean'),
    ('median', 'median'),
    ('std', 'std'),
    ('min', 'min'),
    ('max', 'max')
]).reset_index()

print(desc_stats.round(2))

# Save descriptive statistics
desc_stats.to_sql('descriptive_stats_by_org', conn, if_exists='replace', index=False)
print("[OK] Descriptive statistics saved")

# ===== SIMPLE LINEAR REGRESSION =====
print("\n" + "="*60)
print("SIMPLE LINEAR REGRESSION: Predicting Residents Affected")
print("="*60)

# Prepare data
regression_data = databreach[['total_affected', 'residents_affected']].dropna()
X = regression_data[['total_affected']].values
y = regression_data['residents_affected'].values

# Fit model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Calculate metrics
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print(f"Coefficient (slope): {model.coef_[0]:.6f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"R-squared: {r2:.4f}")
print(f"RMSE: {rmse:.2f}")

# Save regression results
regression_summary = pd.DataFrame({
    'model_type': ['Simple Linear Regression'],
    'dependent_variable': ['residents_affected'],
    'independent_variables': ['total_affected'],
    'coefficient': [str(model.coef_[0])],
    'intercept': [model.intercept_],
    'r_squared': [r2],
    'rmse': [rmse],
    'sample_size': [len(regression_data)]
})
regression_summary.to_sql('simple_regression_results', conn, if_exists='replace', index=False)
print("[OK] Simple regression results saved")

# ===== MULTIPLE REGRESSION =====
print("\n" + "="*60)
print("MULTIPLE REGRESSION: Predicting Total Affected")
print("="*60)

# Create dummy variables for organization type and breach type
multi_reg_data = databreach[['total_affected', 'organization_type', 'breach_type']].dropna()
multi_reg_encoded = pd.get_dummies(multi_reg_data, columns=['organization_type', 'breach_type'], drop_first=True)

X_multi = multi_reg_encoded.drop('total_affected', axis=1)
y_multi = multi_reg_encoded['total_affected']

# Fit multiple regression
multi_model = LinearRegression()
multi_model.fit(X_multi, y_multi)
y_multi_pred = multi_model.predict(X_multi)

# Calculate metrics
r2_multi = r2_score(y_multi, y_multi_pred)
rmse_multi = np.sqrt(mean_squared_error(y_multi, y_multi_pred))

print(f"R-squared: {r2_multi:.4f}")
print(f"RMSE: {rmse_multi:.2f}")
print(f"Number of predictors: {X_multi.shape[1]}")

# Save top coefficients
coef_df = pd.DataFrame({
    'feature': X_multi.columns,
    'coefficient': multi_model.coef_
}).sort_values('coefficient', key=abs, ascending=False)

coef_df.to_sql('multiple_regression_coefficients', conn, if_exists='replace', index=False)

# Save summary
multi_reg_summary = pd.DataFrame({
    'model_type': ['Multiple Linear Regression'],
    'dependent_variable': ['total_affected'],
    'num_predictors': [X_multi.shape[1]],
    'r_squared': [r2_multi],
    'rmse': [rmse_multi],
    'sample_size': [len(multi_reg_data)]
})
multi_reg_summary.to_sql('multiple_regression_results', conn, if_exists='replace', index=False)
print("[OK] Multiple regression results saved")

# ===== REGULARIZED REGRESSION (RIDGE & LASSO) =====
print("\n" + "="*60)
print("REGULARIZED REGRESSION: Ridge and Lasso")
print("="*60)

# Use same data as multiple regression
X_reg = X_multi.copy()
y_reg = y_multi.copy()

# Standardize features (important for regularization)
scaler = StandardScaler()
X_reg_scaled = scaler.fit_transform(X_reg)

# Split data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg_scaled, y_reg, test_size=0.3, random_state=42
)

# Ridge Regression (L2 regularization)
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_reg, y_train_reg)
ridge_train_r2 = ridge_model.score(X_train_reg, y_train_reg)
ridge_test_r2 = ridge_model.score(X_test_reg, y_test_reg)
ridge_pred = ridge_model.predict(X_test_reg)
ridge_rmse = np.sqrt(mean_squared_error(y_test_reg, ridge_pred))

print(f"Ridge Regression (alpha=1.0):")
print(f"  Training R2: {ridge_train_r2:.4f}")
print(f"  Testing R2: {ridge_test_r2:.4f}")
print(f"  Test RMSE: {ridge_rmse:.2f}")

# Lasso Regression (L1 regularization)
lasso_model = Lasso(alpha=100.0, max_iter=10000)
lasso_model.fit(X_train_reg, y_train_reg)
lasso_train_r2 = lasso_model.score(X_train_reg, y_train_reg)
lasso_test_r2 = lasso_model.score(X_test_reg, y_test_reg)
lasso_pred = lasso_model.predict(X_test_reg)
lasso_rmse = np.sqrt(mean_squared_error(y_test_reg, lasso_pred))

# Count non-zero coefficients (Lasso performs feature selection)
non_zero_coefs = np.sum(lasso_model.coef_ != 0)

print(f"\nLasso Regression (alpha=100.0):")
print(f"  Training R2: {lasso_train_r2:.4f}")
print(f"  Testing R2: {lasso_test_r2:.4f}")
print(f"  Test RMSE: {lasso_rmse:.2f}")
print(f"  Non-zero coefficients: {non_zero_coefs}/{len(lasso_model.coef_)}")

# Save results
regularized_results = pd.DataFrame({
    'model_type': ['Ridge', 'Lasso'],
    'alpha': [1.0, 100.0],
    'train_r2': [ridge_train_r2, lasso_train_r2],
    'test_r2': [ridge_test_r2, lasso_test_r2],
    'test_rmse': [ridge_rmse, lasso_rmse],
    'num_features': [X_reg.shape[1], non_zero_coefs],
    'train_size': [len(X_train_reg), len(X_train_reg)],
    'test_size': [len(X_test_reg), len(X_test_reg)]
})
regularized_results.to_sql('regularized_regression_results', conn, if_exists='replace', index=False)
print("[OK] Regularized regression results saved")

# ===== POLYNOMIAL REGRESSION =====
print("\n" + "="*60)
print("POLYNOMIAL REGRESSION: Non-linear Relationships")
print("="*60)

# Use simple relationship: total_affected vs residents_affected
poly_data = databreach[['total_affected', 'residents_affected']].dropna()

# Limit to reasonable range to avoid extreme outliers
poly_data = poly_data[poly_data['total_affected'] < poly_data['total_affected'].quantile(0.95)]

X_poly = poly_data[['total_affected']].values
y_poly = poly_data['residents_affected'].values

# Split data
X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(
    X_poly, y_poly, test_size=0.3, random_state=42
)

# Compare polynomial degrees
poly_results = []
for degree in [1, 2, 3]:
    # Transform features
    poly_features = PolynomialFeatures(degree=degree)
    X_train_poly_transformed = poly_features.fit_transform(X_train_poly)
    X_test_poly_transformed = poly_features.transform(X_test_poly)
    
    # Fit model
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly_transformed, y_train_poly)
    
    # Evaluate
    train_r2 = poly_model.score(X_train_poly_transformed, y_train_poly)
    test_r2 = poly_model.score(X_test_poly_transformed, y_test_poly)
    test_pred = poly_model.predict(X_test_poly_transformed)
    test_rmse = np.sqrt(mean_squared_error(y_test_poly, test_pred))
    
    poly_results.append({
        'degree': degree,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'num_features': X_train_poly_transformed.shape[1]
    })
    
    print(f"Polynomial Degree {degree}:")
    print(f"  Training R2: {train_r2:.4f}")
    print(f"  Testing R2: {test_r2:.4f}")
    print(f"  Test RMSE: {test_rmse:.2f}")

# Save results
poly_results_df = pd.DataFrame(poly_results)
poly_results_df['sample_size'] = len(poly_data)
poly_results_df['train_size'] = len(X_train_poly)
poly_results_df['test_size'] = len(X_test_poly)
poly_results_df.to_sql('polynomial_regression_results', conn, if_exists='replace', index=False)
print("[OK] Polynomial regression results saved")

# ===== LOGISTIC REGRESSION =====
print("\n" + "="*60)
print("LOGISTIC REGRESSION: Predicting Severe Breaches")
print("="*60)

# Define "severe" as breaches affecting > median
median_impact = databreach['total_affected'].median()
logit_data = databreach[['total_affected', 'organization_type', 'breach_type']].dropna()
logit_data['severe'] = (logit_data['total_affected'] > median_impact).astype(int)

# Encode categorical variables
logit_encoded = pd.get_dummies(logit_data[['organization_type', 'breach_type']], drop_first=True)
X_logit = logit_encoded
y_logit = logit_data['severe']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_logit, y_logit, test_size=0.3, random_state=42)

# Fit logistic regression
logit_model = LogisticRegression(max_iter=1000, random_state=42)
logit_model.fit(X_train, y_train)

# Predictions
y_pred_logit = logit_model.predict(X_test)
train_accuracy = logit_model.score(X_train, y_train)
test_accuracy = logit_model.score(X_test, y_test)

print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Testing accuracy: {test_accuracy:.4f}")
print(f"Severe breach threshold: {median_impact:.0f} individuals")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_logit)
print(f"\nConfusion Matrix:")
print(f"  True Negative: {conf_matrix[0,0]}, False Positive: {conf_matrix[0,1]}")
print(f"  False Negative: {conf_matrix[1,0]}, True Positive: {conf_matrix[1,1]}")

# Save logistic regression results
logit_summary = pd.DataFrame({
    'model_type': ['Logistic Regression'],
    'target_variable': ['severe_breach'],
    'threshold': [median_impact],
    'train_accuracy': [train_accuracy],
    'test_accuracy': [test_accuracy],
    'train_size': [len(X_train)],
    'test_size': [len(X_test)]
})
logit_summary.to_sql('logistic_regression_results', conn, if_exists='replace', index=False)
print("[OK] Logistic regression results saved")

# ===== TIME SERIES ANALYSIS =====
print("\n" + "="*60)
print("TIME SERIES ANALYSIS: Breach Trends Over Time")
print("="*60)

# Aggregate breaches by year-month
ts_data = databreach[['breach_date', 'total_affected']].dropna()
ts_data['year_month'] = ts_data['breach_date'].dt.to_period('M')

# Monthly aggregations
monthly_breaches = ts_data.groupby('year_month').agg({
    'breach_date': 'count',
    'total_affected': ['sum', 'mean', 'median']
}).reset_index()

monthly_breaches.columns = ['year_month', 'breach_count', 'total_affected_sum', 
                            'total_affected_mean', 'total_affected_median']
monthly_breaches['year_month'] = monthly_breaches['year_month'].astype(str)

# Save time series data
monthly_breaches.to_sql('time_series_monthly', conn, if_exists='replace', index=False)

print(f"Time period: {monthly_breaches['year_month'].min()} to {monthly_breaches['year_month'].max()}")
print(f"Total months: {len(monthly_breaches)}")
print(f"Average breaches per month: {monthly_breaches['breach_count'].mean():.1f}")
print("[OK] Time series data saved")

# Yearly trends
ts_data['year'] = ts_data['breach_date'].dt.year
yearly_breaches = ts_data.groupby('year').agg({
    'breach_date': 'count',
    'total_affected': ['sum', 'mean']
}).reset_index()
yearly_breaches.columns = ['year', 'breach_count', 'total_affected_sum', 'total_affected_mean']

yearly_breaches.to_sql('time_series_yearly', conn, if_exists='replace', index=False)
print("[OK] Yearly trends saved")

# Close connection
conn.close()

# ===== SUMMARY =====
print("\n" + "="*60)
print("ALL RESULTS SAVED TO DATABASE!")
print("="*60)
print("Tables created:")
print("  1. correlation_results")
print("  2. chi_squared_summary")
print("  3. chi_squared_observed")
print("  4. chi_squared_expected")
print("  5. anova_results")
print("  6. tukey_hsd_results")
print("  7. descriptive_stats_by_org")
print("  8. simple_regression_results")
print("  9. multiple_regression_results")
print(" 10. multiple_regression_coefficients")
print(" 11. regularized_regression_results")
print(" 12. polynomial_regression_results")
print(" 13. logistic_regression_results")
print(" 14. time_series_monthly")
print(" 15. time_series_yearly")
print("="*60)