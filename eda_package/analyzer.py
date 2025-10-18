"""
Breach Analyzer Module
======================

Comprehensive statistical analysis class for data breach exploration.
Wraps existing eda.py functionality in an object-oriented, reusable structure.

Classes:
    BreachAnalyzer: Main analysis class with 15+ statistical methods
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


class BreachAnalyzer:
    """
    Comprehensive statistical analysis toolkit for data breach patterns.
    
    This class provides methods for correlation analysis, hypothesis testing,
    regression modeling, time series analysis, and descriptive statistics.
    All methods from the original eda.py are wrapped in this OOP structure
    for better reusability and documentation.
    
    Attributes:
        df (pd.DataFrame): The breach dataset to analyze
        alpha (float): Significance level for hypothesis testing
        results (dict): Dictionary storing analysis results
        
    Example:
        >>> analyzer = BreachAnalyzer(breach_data, alpha=0.05)
        >>> corr_results = analyzer.correlation_analysis()
        >>> chi_results = analyzer.chi_squared_test('organization_type', 'breach_type')
    """
    
    def __init__(self, df: pd.DataFrame, alpha: float = 0.05):
        """
        Initialize the BreachAnalyzer.
        
        Args:
            df (pd.DataFrame): Breach dataset
            alpha (float): Significance level (default: 0.05)
        """
        self.df = df.copy()
        self.alpha = alpha
        self.results = {}
        
    def correlation_analysis(self, 
                            var1: str = 'total_affected',
                            var2: str = 'residents_affected') -> Dict:
        """
        Perform Pearson and Spearman correlation analysis.
        
        Calculates both linear (Pearson) and monotonic (Spearman) correlations
        to understand the relationship between two continuous variables.
        
        Args:
            var1 (str): First variable name
            var2 (str): Second variable name
            
        Returns:
            dict: Results containing:
                - pearson_r: Pearson correlation coefficient
                - pearson_p: P-value for Pearson test
                - spearman_rho: Spearman correlation coefficient
                - spearman_p: P-value for Spearman test
                - sample_size: Number of valid pairs
                - significant: Boolean indicating significance
                
        Business Interpretation:
            - Pearson measures linear relationship
            - Spearman measures monotonic relationship
            - Large difference suggests non-linear relationship or outliers
        """
        # Get valid pairs
        valid_data = self.df[[var1, var2]].dropna()
        
        # Pearson correlation (linear)
        pearson_r, pearson_p = stats.pearsonr(valid_data[var1], valid_data[var2])
        
        # Spearman correlation (monotonic)
        spearman_rho, spearman_p = stats.spearmanr(valid_data[var1], valid_data[var2])
        
        results = {
            'variable_1': var1,
            'variable_2': var2,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_rho': spearman_rho,
            'spearman_p': spearman_p,
            'sample_size': len(valid_data),
            'pearson_significant': pearson_p < self.alpha,
            'spearman_significant': spearman_p < self.alpha
        }
        
        self.results['correlation'] = results
        return results
    
    def chi_squared_test(self, var1: str, var2: str) -> Dict:
        """
        Test independence between two categorical variables.
        
        Uses chi-squared test to determine if there's a significant relationship
        between two categorical variables (e.g., organization type and breach type).
        
        Args:
            var1 (str): First categorical variable
            var2 (str): Second categorical variable
            
        Returns:
            dict: Results containing:
                - chi2_statistic: Chi-squared test statistic
                - p_value: Statistical significance
                - degrees_of_freedom: df for the test
                - sample_size: Total observations
                - significant: Boolean
                - observed: Contingency table of observed frequencies
                - expected: Expected frequencies under independence
                
        Business Interpretation:
            Significant result means the two variables are related.
            Compare observed vs expected to identify specific patterns.
        """
        # Create contingency table
        contingency_table = pd.crosstab(self.df[var1], self.df[var2])
        
        # Perform chi-squared test
        chi2, p_value, dof, expected_freq = chi2_contingency(contingency_table)
        
        results = {
            'variable_1': var1,
            'variable_2': var2,
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'sample_size': contingency_table.sum().sum(),
            'significant': p_value < self.alpha,
            'observed': contingency_table,
            'expected': pd.DataFrame(expected_freq, 
                                    index=contingency_table.index,
                                    columns=contingency_table.columns)
        }
        
        self.results['chi_squared'] = results
        return results
    
    def anova_test(self, group_var: str, value_var: str) -> Dict:
        """
        Compare means across multiple groups using ANOVA.
        
        Tests if there are significant differences in breach impact across
        different organization types or breach types.
        
        Args:
            group_var (str): Categorical grouping variable
            value_var (str): Continuous variable to compare
            
        Returns:
            dict: Results containing:
                - f_statistic: F-test statistic
                - p_value: Statistical significance
                - groups: Number of groups compared
                - significant: Boolean
                - group_means: Mean for each group
                
        Business Interpretation:
            Significant result means at least one group differs significantly.
            Use post-hoc tests (Tukey HSD) to identify specific differences.
        """
        # Get groups
        groups = [group[value_var].dropna() for name, group in self.df.groupby(group_var)]
        
        # Perform one-way ANOVA
        f_stat, p_value = f_oneway(*groups)
        
        # Calculate group statistics
        group_stats = self.df.groupby(group_var)[value_var].agg(['mean', 'median', 'std', 'count'])
        
        results = {
            'group_variable': group_var,
            'value_variable': value_var,
            'f_statistic': f_stat,
            'p_value': p_value,
            'n_groups': len(groups),
            'significant': p_value < self.alpha,
            'group_statistics': group_stats
        }
        
        self.results['anova'] = results
        return results
    
    def descriptive_statistics(self, group_by: Optional[str] = None) -> pd.DataFrame:
        """
        Generate comprehensive descriptive statistics.
        
        Args:
            group_by (str, optional): Variable to group by (e.g., 'organization_type')
            
        Returns:
            pd.DataFrame: Descriptive statistics including:
                - count, mean, median, std, min, max
                - Q1, Q3, IQR
                - skewness, kurtosis
                
        Business Use:
            Provides overview of data distribution and central tendencies.
            Helps identify outliers and understand typical breach patterns.
        """
        numeric_cols = ['total_affected', 'residents_affected']
        
        if group_by:
            # Group-wise statistics
            stats_df = self.df.groupby(group_by)[numeric_cols[0]].agg([
                'count', 'mean', 'median', 'std', 'min', 'max',
                ('q25', lambda x: x.quantile(0.25)),
                ('q75', lambda x: x.quantile(0.75))
            ]).round(2)
            stats_df['iqr'] = stats_df['q75'] - stats_df['q25']
        else:
            # Overall statistics
            stats_list = []
            for col in numeric_cols:
                data = self.df[col].dropna()
                stats_list.append({
                    'variable': col,
                    'count': len(data),
                    'mean': data.mean(),
                    'median': data.median(),
                    'std': data.std(),
                    'min': data.min(),
                    'max': data.max(),
                    'q25': data.quantile(0.25),
                    'q75': data.quantile(0.75),
                    'skewness': stats.skew(data),
                    'kurtosis': stats.kurtosis(data)
                })
            stats_df = pd.DataFrame(stats_list)
        
        self.results['descriptive_stats'] = stats_df
        return stats_df
    
    def simple_linear_regression(self, 
                                 X_var: str = 'total_affected',
                                 y_var: str = 'residents_affected') -> Dict:
        """
        Perform simple linear regression.
        
        Predicts one variable from another using linear relationship.
        
        Args:
            X_var (str): Predictor variable
            y_var (str): Response variable
            
        Returns:
            dict: Results containing:
                - slope: Regression coefficient
                - intercept: Y-intercept
                - r_squared: Proportion of variance explained
                - predictions: Sample predictions
                
        Business Use:
            Estimate resident impact from total individuals affected.
            R² indicates how well total predicts resident counts.
        """
        # Prepare data
        valid_data = self.df[[X_var, y_var]].dropna()
        X = valid_data[[X_var]].values
        y = valid_data[y_var].values
        
        # Fit model
        model = LinearRegression()
        model.fit(X, y)
        
        # Get predictions and R²
        y_pred = model.predict(X)
        r_squared = model.score(X, y)
        
        results = {
            'X_variable': X_var,
            'y_variable': y_var,
            'slope': model.coef_[0],
            'intercept': model.intercept_,
            'r_squared': r_squared,
            'sample_size': len(valid_data),
            'model': model
        }
        
        self.results['simple_regression'] = results
        return results
    
    def time_series_analysis(self, date_col: str = 'breach_date',
                            freq: str = 'Y') -> pd.DataFrame:
        """
        Analyze breach trends over time.
        
        Args:
            date_col (str): Date column to analyze
            freq (str): Frequency for grouping ('Y' for year, 'M' for month)
            
        Returns:
            pd.DataFrame: Time series summary with:
                - breach_count: Number of breaches per period
                - total_affected_sum: Total individuals affected
                - total_affected_mean: Average impact per breach
                
        Business Use:
            Identify trends in breach frequency and severity over time.
            Support forecasting and resource planning.
        """
        df_time = self.df[[date_col, 'total_affected']].copy()
        df_time = df_time.dropna(subset=[date_col])
        
        # Extract time period
        if freq == 'Y':
            df_time['period'] = df_time[date_col].dt.year
        elif freq == 'M':
            df_time['period'] = df_time[date_col].dt.to_period('M')
        else:
            raise ValueError("freq must be 'Y' or 'M'")
        
        # Aggregate
        time_series = df_time.groupby('period').agg({
            date_col: 'count',
            'total_affected': ['sum', 'mean']
        })
        
        time_series.columns = ['breach_count', 'total_affected_sum', 'total_affected_mean']
        time_series = time_series.reset_index()
        
        self.results['time_series'] = time_series
        return time_series
    
    def logistic_regression_severity(self, threshold: float = 10000) -> Dict:
        """
        Predict severe vs non-severe breaches using logistic regression.
        
        Classifies breaches as 'severe' if total_affected exceeds threshold.
        
        Args:
            threshold (float): Cutoff for severe classification
            
        Returns:
            dict: Results containing:
                - accuracy: Model accuracy
                - coefficients: Feature coefficients
                - severe_count: Number of severe breaches
                - model: Trained model
                
        Business Use:
            Identify factors that predict high-impact breaches.
            Prioritize prevention efforts for severe incidents.
        """
        # Prepare data
        df_model = self.df[['organization_type', 'breach_type', 'total_affected']].copy()
        df_model = df_model.dropna()
        
        # Create binary target
        df_model['severe'] = (df_model['total_affected'] > threshold).astype(int)
        
        # One-hot encode categorical variables
        X = pd.get_dummies(df_model[['organization_type', 'breach_type']], 
                          drop_first=True)
        y = df_model['severe']
        
        # Fit model
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        
        # Calculate accuracy
        accuracy = model.score(X, y)
        
        results = {
            'threshold': threshold,
            'accuracy': accuracy,
            'severe_count': y.sum(),
            'non_severe_count': len(y) - y.sum(),
            'feature_names': X.columns.tolist(),
            'coefficients': dict(zip(X.columns, model.coef_[0])),
            'model': model
        }
        
        self.results['logistic_regression'] = results
        return results
    
    def get_business_insights(self) -> Dict[str, str]:
        """
        Generate business-focused insights from all analyses.
        
        Returns:
            dict: Key insights for business stakeholders
            
        Business Use:
            Provides executive summary of findings.
            Supports strategic decision-making and resource allocation.
        """
        insights = {}
        
        # Correlation insights
        if 'correlation' in self.results:
            corr = self.results['correlation']
            insights['impact_relationship'] = (
                f"Total and resident impact show "
                f"{'strong' if abs(corr['spearman_rho']) > 0.7 else 'moderate' if abs(corr['spearman_rho']) > 0.4 else 'weak'} "
                f"relationship (Spearman rho = {corr['spearman_rho']:.3f}). "
                f"{'Non-linear pattern detected.' if abs(corr['spearman_rho']) - abs(corr['pearson_r']) > 0.15 else ''}"
            )
        
        # Chi-squared insights
        if 'chi_squared' in self.results:
            chi = self.results['chi_squared']
            insights['industry_vulnerability'] = (
                f"Organization type and breach type are "
                f"{'significantly' if chi['significant'] else 'not'} related "
                f"(χ² = {chi['chi2_statistic']:.2f}, p {'<' if chi['p_value'] < 0.001 else '='} "
                f"{chi['p_value']:.3f}). Different industries face different threats."
            )
        
        # ANOVA insights
        if 'anova' in self.results:
            anova = self.results['anova']
            insights['impact_variation'] = (
                f"Breach impact varies "
                f"{'significantly' if anova['significant'] else 'not significantly'} "
                f"across {'groups' if anova['n_groups'] > 2 else 'categories'} "
                f"(F = {anova['f_statistic']:.2f}, p = {anova['p_value']:.3f})."
            )
        
        return insights
    
    def summary_report(self) -> str:
        """
        Generate comprehensive text summary of all analyses.
        
        Returns:
            str: Formatted report summarizing key findings
        """
        report = "DATA BREACH ANALYSIS SUMMARY\n"
        report += "=" * 60 + "\n\n"
        
        for analysis, results in self.results.items():
            report += f"{analysis.upper().replace('_', ' ')}\n"
            report += "-" * 60 + "\n"
            report += str(results) + "\n\n"
        
        return report
    
    def __repr__(self):
        """String representation."""
        return f"BreachAnalyzer(records={len(self.df)}, alpha={self.alpha}, analyses={len(self.results)})"
