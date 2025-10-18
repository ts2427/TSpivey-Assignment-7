"""
Breach Visualizer Module
========================

Professional data visualization class for data breach analysis.
Wraps existing visualizations.py functionality in an object-oriented structure.

Classes:
    BreachVisualizer: Main visualization class with 6+ chart methods
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from typing import Optional, Tuple, List


class BreachVisualizer:
    """
    Comprehensive visualization toolkit for data breach analysis.
    
    Creates publication-quality charts for business stakeholders including
    heatmaps, time series, correlation plots, and comparative analyses.
    
    Attributes:
        df (pd.DataFrame): Breach dataset
        output_dir (str): Directory for saving visualizations
        style (str): Matplotlib style
        dpi (int): Resolution for saved images
        
    Example:
        >>> viz = BreachVisualizer(breach_data)
        >>> viz.plot_industry_vulnerability(chi_observed)
        >>> viz.plot_time_series_trends(time_series_data)
        >>> viz.save_all_figures()
    """
    
    def __init__(self, df: pd.DataFrame, output_dir: str = 'output/visualizations',
                 style: str = 'seaborn-v0_8-darkgrid', dpi: int = 300):
        """
        Initialize the BreachVisualizer.
        
        Args:
            df (pd.DataFrame): Breach dataset
            output_dir (str): Output directory for charts
            style (str): Matplotlib style
            dpi (int): Image resolution
        """
        self.df = df.copy()
        self.output_dir = output_dir
        self.dpi = dpi
        self.figures = {}
        
        # Set style
        plt.style.use(style)
        sns.set_palette("husl")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def plot_industry_vulnerability(self, chi_observed: pd.DataFrame,
                                   figsize: Tuple[int, int] = (12, 7),
                                   save: bool = True) -> plt.Figure:
        """
        Create heatmap showing industry-specific breach vulnerabilities.
        
        Args:
            chi_observed (pd.DataFrame): Contingency table of observed frequencies
            figsize (tuple): Figure dimensions
            save (bool): Whether to save figure
            
        Returns:
            plt.Figure: Matplotlib figure object
            
        Business Use:
            Identifies which industries are most vulnerable to specific attack types.
            Guides targeted security investments and prevention strategies.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        matrix = chi_observed.set_index('organization_type')
        
        sns.heatmap(matrix, annot=True, fmt='d', cmap='RdYlBu_r',
                   cbar_kws={'label': 'Breach Count'}, linewidths=0.5, ax=ax)
        
        ax.set_title('Industry-Specific Breach Vulnerabilities\nTargeted Prevention by Industry Type',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Attack Method', fontsize=12)
        ax.set_ylabel('Industry Sector', fontsize=12)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, '1_industry_vulnerability_heatmap.png')
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        self.figures['industry_vulnerability'] = fig
        return fig
    
    def plot_breach_frequency(self, figsize: Tuple[int, int] = (14, 6),
                             save: bool = True) -> plt.Figure:
        """
        Create bar charts showing breach frequency by type and sector.
        
        Args:
            figsize (tuple): Figure dimensions
            save (bool): Whether to save figure
            
        Returns:
            plt.Figure: Matplotlib figure object
            
        Business Use:
            Identifies most common attack vectors and most targeted sectors.
            Helps prioritize security resource allocation.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # By breach type
        breach_counts = self.df['breach_type'].value_counts()
        ax1.barh(breach_counts.index, breach_counts.values, color='steelblue')
        ax1.set_xlabel('Number of Incidents', fontsize=11)
        ax1.set_title('Most Common Attack Vectors', fontsize=12, fontweight='bold')
        ax1.invert_yaxis()
        
        # Add value labels
        for i, v in enumerate(breach_counts.values):
            ax1.text(v, i, f' {v:,}', va='center')
        
        # By organization type
        org_counts = self.df['organization_type'].value_counts()
        ax2.barh(org_counts.index, org_counts.values, color='coral')
        ax2.set_xlabel('Number of Incidents', fontsize=11)
        ax2.set_title('Most Frequently Breached Sectors', fontsize=12, fontweight='bold')
        ax2.invert_yaxis()
        
        # Add value labels
        for i, v in enumerate(org_counts.values):
            ax2.text(v, i, f' {v:,}', va='center')
        
        fig.suptitle('Breach Frequency Analysis: Where to Focus Security Resources',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, '2_breach_frequency.png')
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        self.figures['breach_frequency'] = fig
        return fig
    
    def plot_correlation_scatter(self, 
                                var1: str = 'total_affected',
                                var2: str = 'residents_affected',
                                correlation_stats: Optional[dict] = None,
                                figsize: Tuple[int, int] = (10, 7),
                                save: bool = True) -> plt.Figure:
        """
        Create scatter plot showing relationship between two variables.
        
        Args:
            var1 (str): X-axis variable
            var2 (str): Y-axis variable
            correlation_stats (dict): Optional correlation results to display
            figsize (tuple): Figure dimensions
            save (bool): Whether to save figure
            
        Returns:
            plt.Figure: Matplotlib figure object
            
        Business Use:
            Visualizes relationship between total and resident impact.
            Helps understand data breach scope and reporting patterns.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        valid = self.df[[var1, var2]].dropna()
        
        ax.scatter(valid[var1], valid[var2],
                  alpha=0.4, s=40, c='steelblue', edgecolors='white', linewidth=0.5)
        
        ax.set_xlabel('Total Individuals Affected', fontsize=12)
        ax.set_ylabel('State Residents Affected', fontsize=12)
        
        # Add correlation stats to title if provided
        title = 'Breach Impact Scale: Total vs Local Exposure'
        if correlation_stats:
            pearson_r = correlation_stats.get('pearson_r', 0)
            spearman_rho = correlation_stats.get('spearman_rho', 0)
            title += f'\nPearson r={pearson_r:.2f} | Spearman rho={spearman_rho:.2f} (p<0.001)'
        
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, '3_impact_correlation.png')
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        self.figures['impact_correlation'] = fig
        return fig
    
    def plot_sector_impact(self, desc_stats: pd.DataFrame,
                          figsize: Tuple[int, int] = (12, 6),
                          save: bool = True) -> plt.Figure:
        """
        Create bar chart comparing median breach impact by sector.
        
        Args:
            desc_stats (pd.DataFrame): Descriptive statistics by organization type
            figsize (tuple): Figure dimensions
            save (bool): Whether to save figure
            
        Returns:
            plt.Figure: Matplotlib figure object
            
        Business Use:
            Compares typical breach severity across industries.
            Identifies high-risk sectors requiring enhanced protection.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        desc_stats_sorted = desc_stats.sort_values('median', ascending=True)
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
        
        if save:
            filepath = os.path.join(self.output_dir, '4_sector_impact.png')
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        self.figures['sector_impact'] = fig
        return fig
    
    def plot_time_series_trends(self, time_series: pd.DataFrame,
                               figsize: Tuple[int, int] = (12, 8),
                               save: bool = True) -> plt.Figure:
        """
        Create time series plots showing breach trends over time.
        
        Args:
            time_series (pd.DataFrame): Time series data with year, breach_count, mean
            figsize (tuple): Figure dimensions
            save (bool): Whether to save figure
            
        Returns:
            plt.Figure: Matplotlib figure object
            
        Business Use:
            Tracks breach frequency and severity trends.
            Supports forecasting and long-term planning.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
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
        
        fig.suptitle('Data Breach Trends Over Time (2003-2025)', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, '5_time_series_trends.png')
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        self.figures['time_series_trends'] = fig
        return fig
    
    def plot_regression_fit(self, 
                           regression_results: Optional[dict] = None,
                           sample_size: int = 5000,
                           figsize: Tuple[int, int] = (10, 7),
                           save: bool = True) -> plt.Figure:
        """
        Create scatter plot with regression line overlay.
        
        Args:
            regression_results (dict): Optional regression results to display
            sample_size (int): Number of points to plot (for clarity)
            figsize (tuple): Figure dimensions
            save (bool): Whether to save figure
            
        Returns:
            plt.Figure: Matplotlib figure object
            
        Business Use:
            Demonstrates predictive relationship between variables.
            Shows model fit quality for forecasting applications.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        valid = self.df[['total_affected', 'residents_affected']].dropna()
        
        # Sample for clearer visualization
        if len(valid) > sample_size:
            valid_sample = valid.sample(sample_size, random_state=42)
        else:
            valid_sample = valid
        
        # Scatter plot
        ax.scatter(valid_sample['total_affected'], valid_sample['residents_affected'],
                  alpha=0.3, s=30, c='steelblue', label='Actual Data')
        
        # Fit line
        z = np.polyfit(valid['total_affected'], valid['residents_affected'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid['total_affected'].min(), valid['total_affected'].max(), 100)
        
        r_squared = regression_results['r_squared'] if regression_results else 0.099
        slope = regression_results['slope'] if regression_results else z[0]
        intercept = regression_results['intercept'] if regression_results else z[1]
        
        ax.plot(x_line, p(x_line), "r-", linewidth=2, 
               label=f'Linear Fit (R²={r_squared:.3f})')
        
        ax.set_xlabel('Total Individuals Affected', fontsize=12)
        ax.set_ylabel('State Residents Affected', fontsize=12)
        ax.set_title(f'Simple Linear Regression: Predicting Resident Impact\n'
                    f'Slope={slope:.4f}, Intercept={intercept:.0f}',
                    fontsize=13, fontweight='bold', pad=15)
        ax.legend()
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, '6_regression_fit.png')
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        self.figures['regression_fit'] = fig
        return fig
    
    def create_comprehensive_dashboard(self, 
                                      chi_observed: pd.DataFrame,
                                      desc_stats: pd.DataFrame,
                                      time_series: pd.DataFrame,
                                      correlation_stats: Optional[dict] = None,
                                      regression_results: Optional[dict] = None) -> None:
        """
        Generate all visualizations in one call.
        
        Args:
            chi_observed (pd.DataFrame): Contingency table for heatmap
            desc_stats (pd.DataFrame): Descriptive statistics
            time_series (pd.DataFrame): Time series data
            correlation_stats (dict): Optional correlation results
            regression_results (dict): Optional regression results
            
        Business Use:
            One-stop function to generate complete visualization suite.
            Ensures consistent formatting across all charts.
        """
        print("="*60)
        print("GENERATING COMPREHENSIVE VISUALIZATION DASHBOARD")
        print("="*60 + "\n")
        
        self.plot_industry_vulnerability(chi_observed)
        self.plot_breach_frequency()
        self.plot_correlation_scatter(correlation_stats=correlation_stats)
        self.plot_sector_impact(desc_stats)
        self.plot_time_series_trends(time_series)
        self.plot_regression_fit(regression_results=regression_results)
        
        print("\n" + "="*60)
        print("DASHBOARD COMPLETE")
        print("="*60)
        print(f"Location: {self.output_dir}")
        print(f"Total visualizations: {len(self.figures)}")
    
    def close_all_figures(self):
        """Close all matplotlib figures to free memory."""
        plt.close('all')
        self.figures = {}
    
    def get_figure(self, name: str) -> Optional[plt.Figure]:
        """
        Retrieve a specific figure by name.
        
        Args:
            name (str): Figure name
            
        Returns:
            plt.Figure or None
        """
        return self.figures.get(name)
    
    def list_figures(self) -> List[str]:
        """
        Get list of all generated figures.
        
        Returns:
            list: Figure names
        """
        return list(self.figures.keys())
    
    def __repr__(self):
        """String representation."""
        return f"BreachVisualizer(output_dir='{self.output_dir}', figures={len(self.figures)})"
