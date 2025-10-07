"""
Data Visualization Module
Creates business-focused visualizations from breach data analysis
"""

import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def create_output_directory():
    """Create directory for visualizations"""
    if not os.path.exists('output/visualizations'):
        os.makedirs('output/visualizations')

def load_data():
    """Load data from database"""
    conn = sqlite3.connect('databreach.db')
    databreach = pd.read_sql_query("SELECT * FROM databreach", conn)
    chi_observed = pd.read_sql_query("SELECT * FROM chi_squared_observed", conn)
    time_series = pd.read_sql_query("SELECT * FROM time_series_yearly", conn)
    desc_stats = pd.read_sql_query("SELECT * FROM descriptive_stats_by_org", conn)
    conn.close()
    
    # Convert numeric columns
    databreach['total_affected'] = pd.to_numeric(databreach['total_affected'], errors='coerce')
    databreach['residents_affected'] = pd.to_numeric(databreach['residents_affected'], errors='coerce')
    
    return databreach, chi_observed, time_series, desc_stats

def plot_industry_vulnerability(chi_observed):
    """Heatmap: Industry-specific breach vulnerabilities"""
    plt.figure(figsize=(12, 7))
    matrix = chi_observed.set_index('organization_type')
    
    sns.heatmap(matrix, annot=True, fmt='d', cmap='RdYlBu_r',
                cbar_kws={'label': 'Breach Count'}, linewidths=0.5)
    
    plt.title('Industry-Specific Breach Vulnerabilities\nTarget Prevention by Industry Type',
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Attack Method', fontsize=12)
    plt.ylabel('Industry Sector', fontsize=12)
    plt.tight_layout()
    plt.savefig('output/visualizations/1_industry_vulnerability_heatmap.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: 1_industry_vulnerability_heatmap.png")

def plot_breach_frequency(databreach):
    """Bar charts: Most common attack vectors and sectors"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # By breach type
    breach_counts = databreach['breach_type'].value_counts()
    ax1.barh(breach_counts.index, breach_counts.values, color='steelblue')
    ax1.set_xlabel('Number of Incidents', fontsize=11)
    ax1.set_title('Most Common Attack Vectors', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    
    # Add value labels
    for i, v in enumerate(breach_counts.values):
        ax1.text(v, i, f' {v:,}', va='center')
    
    # By organization type
    org_counts = databreach['organization_type'].value_counts()
    ax2.barh(org_counts.index, org_counts.values, color='coral')
    ax2.set_xlabel('Number of Incidents', fontsize=11)
    ax2.set_title('Most Frequently Breached Sectors', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    
    # Add value labels
    for i, v in enumerate(org_counts.values):
        ax2.text(v, i, f' {v:,}', va='center')
    
    plt.suptitle('Breach Frequency Analysis: Where to Focus Security Resources',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('output/visualizations/2_breach_frequency.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: 2_breach_frequency.png")

def plot_correlation_scatter(databreach):
    """Scatter plot: Total vs residents affected"""
    valid = databreach[['total_affected', 'residents_affected']].dropna()
    
    plt.figure(figsize=(10, 7))
    plt.scatter(valid['total_affected'], valid['residents_affected'],
                alpha=0.4, s=40, c='steelblue', edgecolors='white', linewidth=0.5)
    
    plt.xlabel('Total Individuals Affected', fontsize=12)
    plt.ylabel('State Residents Affected', fontsize=12)
    plt.title('Breach Impact Scale: Total vs Local Exposure\nPearson r=0.32 | Spearman rho=0.52 (p<0.001)',
              fontsize=13, fontweight='bold', pad=15)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('output/visualizations/3_impact_correlation.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: 3_impact_correlation.png")

def plot_sector_impact(desc_stats):
    """Bar chart: Median breach impact by sector"""
    desc_stats_sorted = desc_stats.sort_values('median', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(desc_stats_sorted)))
    
    ax.barh(desc_stats_sorted['organization_type'], 
            desc_stats_sorted['median'], 
            color=colors)
    
    ax.set_xlabel('Number of Individuals Affected (Median)', fontsize=11)
    ax.set_xscale('log')
    ax.set_title('Breach Impact by Industry Sector\nMedian Individuals Affected per Incident',
                 fontsize=13, fontweight='bold', pad=15)
    
    # Add value labels
    for i, (idx, row) in enumerate(desc_stats_sorted.iterrows()):
        ax.text(row['median'], i, f" {row['median']:.0f}", va='center')
    
    plt.tight_layout()
    plt.savefig('output/visualizations/4_sector_impact.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: 4_sector_impact.png")

def plot_time_series(time_series):
    """Line chart: Breaches over time"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Breach count over time
    ax1.plot(time_series['year'], time_series['breach_count'], 
             marker='o', linewidth=2, markersize=6, color='steelblue')
    ax1.set_xlabel('Year', fontsize=11)
    ax1.set_ylabel('Number of Breaches', fontsize=11)
    ax1.set_title('Annual Breach Frequency Trend', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Average impact over time
    ax2.plot(time_series['year'], time_series['total_affected_mean'], 
             marker='s', linewidth=2, markersize=6, color='coral')
    ax2.set_xlabel('Year', fontsize=11)
    ax2.set_ylabel('Average Individuals Affected', fontsize=11)
    ax2.set_title('Average Breach Impact Trend', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Data Breach Trends Over Time (2003-2025)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('output/visualizations/5_time_series_trends.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: 5_time_series_trends.png")

def plot_regression_fit(databreach):
    """Scatter with regression line"""
    valid = databreach[['total_affected', 'residents_affected']].dropna()
    
    # Sample data to make visualization clearer
    if len(valid) > 5000:
        valid_sample = valid.sample(5000, random_state=42)
    else:
        valid_sample = valid
    
    plt.figure(figsize=(10, 7))
    
    # Scatter plot
    plt.scatter(valid_sample['total_affected'], valid_sample['residents_affected'],
                alpha=0.3, s=30, c='steelblue', label='Actual Data')
    
    # Fit line
    z = np.polyfit(valid['total_affected'], valid['residents_affected'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid['total_affected'].min(), valid['total_affected'].max(), 100)
    plt.plot(x_line, p(x_line), "r-", linewidth=2, label=f'Linear Fit (R²=0.099)')
    
    plt.xlabel('Total Individuals Affected', fontsize=12)
    plt.ylabel('State Residents Affected', fontsize=12)
    plt.title('Simple Linear Regression: Predicting Resident Impact\nSlope=0.0027, Intercept=3193',
              fontsize=13, fontweight='bold', pad=15)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('output/visualizations/6_regression_fit.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: 6_regression_fit.png")

def generate_all_visualizations():
    """Main function to generate all visualizations"""
    print("="*60)
    print("GENERATING DATA VISUALIZATIONS")
    print("="*60)
    
    create_output_directory()
    databreach, chi_observed, time_series, desc_stats = load_data()
    
    print("\nCreating visualizations...")
    plot_industry_vulnerability(chi_observed)
    plot_breach_frequency(databreach)
    plot_correlation_scatter(databreach)
    plot_sector_impact(desc_stats)
    plot_time_series(time_series)
    plot_regression_fit(databreach)
    
    print("\n" + "="*60)
    print("ALL VISUALIZATIONS COMPLETE")
    print("="*60)
    print("Location: output/visualizations/")
    print("Total files: 6 PNG images")
    print("\nFiles created:")
    print("  1. Industry vulnerability heatmap")
    print("  2. Breach frequency analysis")
    print("  3. Impact correlation scatter")
    print("  4. Sector impact comparison")
    print("  5. Time series trends")
    print("  6. Regression fit visualization")

if __name__ == "__main__":
    generate_all_visualizations()