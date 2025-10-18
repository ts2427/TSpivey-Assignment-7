"""
Data Loader Module
==================

Provides object-oriented interface for loading data from the data breach database.
Wraps existing dataload.py functionality in a reusable class structure.

Classes:
    DataLoader: Main class for database operations
"""

import pandas as pd
import sqlite3
from typing import Optional, Dict, List


class DataLoader:
    """
    Manages database connections and data loading for breach analysis.
    
    This class provides a clean interface for accessing breach data, analytical results,
    and SEC company reference information from the SQLite database.
    
    Attributes:
        db_name (str): Path to the SQLite database file
        connection: Active database connection (when connected)
        
    Example:
        >>> loader = DataLoader('databreach.db')
        >>> df = loader.load_breach_data()
        >>> print(f"Loaded {len(df)} breach records")
    """
    
    def __init__(self, db_name: str = 'databreach.db'):
        """
        Initialize the DataLoader.
        
        Args:
            db_name (str): Path to SQLite database file
        """
        self.db_name = db_name
        self.connection = None
        
    def connect(self) -> sqlite3.Connection:
        """
        Establish connection to the database.
        
        Returns:
            sqlite3.Connection: Active database connection
        """
        self.connection = sqlite3.connect(self.db_name)
        return self.connection
    
    def disconnect(self):
        """Close the database connection if open."""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    def load_breach_data(self) -> pd.DataFrame:
        """
        Load the main breach dataset.
        
        Returns:
            pd.DataFrame: Complete breach dataset with all 20 columns
            
        Note:
            Automatically converts date columns to datetime and numeric columns
            to appropriate types.
        """
        conn = self.connect()
        df = pd.read_sql_query("SELECT * FROM databreach", conn)
        self.disconnect()
        
        # Convert date columns
        date_columns = ['reported_date', 'breach_date', 'end_breach_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Convert numeric columns
        numeric_columns = ['total_affected', 'residents_affected']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def load_statistical_results(self, table_name: str) -> pd.DataFrame:
        """
        Load results from a specific statistical analysis table.
        
        Args:
            table_name (str): Name of the statistical results table
            
        Returns:
            pd.DataFrame: Statistical results
            
        Available tables:
            - correlation_results
            - chi_squared_summary
            - chi_squared_observed
            - chi_squared_expected
            - anova_results
            - tukey_hsd_results
            - descriptive_stats_by_org
            - simple_regression_results
            - multiple_regression_results
            - time_series_monthly
            - time_series_yearly
        """
        conn = self.connect()
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        self.disconnect()
        return df
    
    def load_sec_reference(self) -> pd.DataFrame:
        """
        Load SEC company reference data.
        
        Returns:
            pd.DataFrame: SEC company information including CIK, ticker, exchange
        """
        conn = self.connect()
        df = pd.read_sql_query("SELECT * FROM sec_company_reference", conn)
        self.disconnect()
        return df
    
    def get_table_info(self) -> Dict[str, int]:
        """
        Get row counts for all tables in the database.
        
        Returns:
            dict: Dictionary mapping table names to row counts
            
        Example:
            >>> loader = DataLoader()
            >>> info = loader.get_table_info()
            >>> print(f"Breach records: {info['databreach']:,}")
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        table_info = {}
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            table_info[table_name] = count
        
        self.disconnect()
        return table_info
    
    def list_available_tables(self) -> List[str]:
        """
        Get list of all available tables in the database.
        
        Returns:
            list: Table names
        """
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cursor.fetchall()]
        self.disconnect()
        return tables
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute a custom SQL query.
        
        Args:
            query (str): SQL query string
            
        Returns:
            pd.DataFrame: Query results
            
        Example:
            >>> loader = DataLoader()
            >>> df = loader.execute_query(
            ...     "SELECT organization_type, COUNT(*) as count "
            ...     "FROM databreach GROUP BY organization_type"
            ... )
        """
        conn = self.connect()
        df = pd.read_sql_query(query, conn)
        self.disconnect()
        return df
    
    def get_breach_summary(self) -> Dict:
        """
        Get high-level summary statistics about the breach dataset.
        
        Returns:
            dict: Summary statistics including record counts, date ranges, etc.
        """
        df = self.load_breach_data()
        
        summary = {
            'total_records': len(df),
            'date_range': {
                'earliest': df['breach_date'].min(),
                'latest': df['breach_date'].max()
            },
            'organization_types': df['organization_type'].nunique(),
            'breach_types': df['breach_type'].nunique(),
            'total_individuals_affected': df['total_affected'].sum(),
            'records_with_impact_data': df['total_affected'].notna().sum(),
            'missing_data_percentage': (df.isnull().sum() / len(df) * 100).to_dict()
        }
        
        return summary
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
    
    def __repr__(self):
        """String representation."""
        return f"DataLoader(db_name='{self.db_name}')"
