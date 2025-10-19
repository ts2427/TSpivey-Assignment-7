"""
Feature Engineering Module
===========================

Transforms raw breach data into ML-ready features based on insights
from exploratory data analysis (Assignment 5).

Classes:
    FeatureEngineer: Main feature engineering pipeline
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional


class FeatureEngineer:
    """
    Feature engineering pipeline for breach prediction models.
    
    Based on EDA findings from Assignment 5:
    - Organization type significantly impacts breach patterns (chi-squared p<0.001)
    - Breach type varies by industry (chi-squared analysis)
    - Temporal patterns exist in breach frequency
    - Non-linear relationships in impact metrics
    
    Attributes:
        scaler: StandardScaler for numeric features
        feature_names: List of feature column names after encoding
        
    Example:
        >>> engineer = FeatureEngineer()
        >>> X_train, X_test, y_train, y_test = engineer.prepare_data(df, target='is_severe')
        >>> print(f"Training set: {X_train.shape}")
    """
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def create_target_variable(self, df: pd.DataFrame, 
                               threshold: int = 1000,
                               target_column: str = 'total_affected') -> pd.Series:
        """
        Create binary severity classification target.
        
        Based on business context: breaches affecting >10,000 individuals
        are considered "severe" and require different response protocols.
        
        Args:
            df: Breach dataframe
            threshold: Number of affected individuals for "severe" classification
            target_column: Column to use for threshold
            
        Returns:
            pd.Series: Binary target (1=severe, 0=non-severe)
            
        Example:
            >>> y = engineer.create_target_variable(df, threshold=1000)
            >>> print(f"Severe breaches: {y.sum()} of {len(y)}")
        """
        return (df[target_column] > threshold).astype(int)
    
    def engineer_temporal_features(self, df: pd.DataFrame, 
                                   date_column: str = 'breach_date') -> pd.DataFrame:
        """
        Extract temporal features from breach dates.
        
        Time-based patterns identified in EDA:
        - Year: Breach frequency trends over time
        - Month: Seasonal patterns
        - Quarter: Quarterly business cycle effects
        - Day of week: Operational timing patterns
        
        Args:
            df: Breach dataframe
            date_column: Name of date column
            
        Returns:
            pd.DataFrame: Original df with new temporal features
        """
        df = df.copy()
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        
        # Extract temporal features
        df['breach_year'] = df[date_column].dt.year
        df['breach_month'] = df[date_column].dt.month
        df['breach_quarter'] = df[date_column].dt.quarter
        df['breach_day_of_week'] = df[date_column].dt.dayofweek
        df['is_weekend'] = (df['breach_day_of_week'] >= 5).astype(int)
        
        return df
    
    def engineer_categorical_features(self, df: pd.DataFrame,
                                     categorical_columns: List[str]) -> pd.DataFrame:
        """
        One-hot encode categorical features.
        
        Based on EDA chi-squared analysis showing significant relationships
        between organization type, breach type, and breach outcomes.
        
        Args:
            df: Breach dataframe
            categorical_columns: Columns to encode
            
        Returns:
            pd.DataFrame: Dataframe with one-hot encoded features
        """
        df_encoded = pd.get_dummies(df, columns=categorical_columns, 
                                    drop_first=True, dtype=int)
        return df_encoded
    
    def select_features(self, df: pd.DataFrame,
                       target_column: str = 'is_severe') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Select features for modeling based on EDA insights.
        
        Feature selection based on Assignment 5 findings:
        - Organization type: Significant predictor (ANOVA p=0.010)
        - Breach type: Strongly related to organization (chi-squared p<0.001)
        - Temporal features: Time series patterns observed
        
        Args:
            df: Engineered features dataframe
            target_column: Name of target variable
            
        Returns:
            Tuple of (X, y): Features and target
        """
        # Define feature categories
        categorical_features = ['organization_type', 'breach_type']
        temporal_features = ['breach_year', 'breach_month', 'breach_quarter', 
                            'breach_day_of_week', 'is_weekend']
        
        # Check which features exist
        available_features = []
        for feat in categorical_features + temporal_features:
            if feat in df.columns:
                available_features.append(feat)
        
        # Also include any engineered categorical columns (one-hot encoded)
        for col in df.columns:
            if (col.startswith('organization_type_') or 
                col.startswith('breach_type_') or
                col in temporal_features):
                if col not in available_features:
                    available_features.append(col)
        
        X = df[available_features]
        y = df[target_column]
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def prepare_data(self, df: pd.DataFrame,
                    target_column: str = 'is_severe',
                    threshold: int = 1000,
                    test_size: float = 0.2,
                    random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                      pd.Series, pd.Series]:
        """
        Complete feature engineering pipeline.
        
        Steps:
        1. Create target variable (if not exists)
        2. Engineer temporal features
        3. Handle missing values
        4. Encode categorical features
        5. Select features
        6. Split train/test
        
        Args:
            df: Raw breach dataframe
            target_column: Name of target variable
            threshold: Severity threshold for classification
            test_size: Proportion for test set
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
            
        Example:
            >>> from eda_package import DataLoader
            >>> loader = DataLoader('databreach.db')
            >>> df = loader.load_breach_data()
            >>> 
            >>> engineer = FeatureEngineer()
            >>> X_train, X_test, y_train, y_test = engineer.prepare_data(df)
            >>> print(f"Ready for modeling: {X_train.shape[0]} training samples")
        """
        df = df.copy()
        
        # Step 1: Create target if needed
        if target_column not in df.columns:
            df[target_column] = self.create_target_variable(df, threshold)
        
        # Step 2: Engineer temporal features
        if 'breach_date' in df.columns:
            df = self.engineer_temporal_features(df)
        
        # Step 3: Handle missing values in key columns
        # Drop rows missing critical features
        required_columns = ['organization_type', 'breach_type', target_column]
        df = df.dropna(subset=required_columns)
        
        # Step 4: Encode categorical features
        categorical_columns = ['organization_type', 'breach_type']
        df = self.engineer_categorical_features(df, categorical_columns)
        
        # Step 5: Select features
        X, y = self.select_features(df, target_column)
        
        # Step 5.5: Handle any remaining NaN values
        # Fill NaN with 0 (safe for one-hot encoded and temporal features)
        X = X.fillna(0)
        
        # Step 6: Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Feature Engineering Complete:")
        print(f"  Training samples: {X_train.shape[0]:,}")
        print(f"  Test samples: {X_test.shape[0]:,}")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Severe breaches in training: {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
        
        return X_train, X_test, y_train, y_test
    
    def prepare_regression_data(self, df: pd.DataFrame,
                               target_column: str = 'total_affected',
                               test_size: float = 0.2,
                               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame,
                                                                pd.Series, pd.Series]:
        """
        Prepare data for regression (predicting continuous impact values).
        
        Similar to prepare_data() but for regression instead of classification.
        Predicts actual number of individuals affected rather than binary severity.
        
        Args:
            df: Raw breach dataframe
            target_column: Continuous target (e.g., 'total_affected')
            test_size: Proportion for test set
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        df = df.copy()
        
        # Engineer temporal features
        if 'breach_date' in df.columns:
            df = self.engineer_temporal_features(df)
        
        # Drop rows with missing target
        df = df.dropna(subset=[target_column, 'organization_type', 'breach_type'])
        
        # Encode categorical features
        categorical_columns = ['organization_type', 'breach_type']
        df = self.engineer_categorical_features(df, categorical_columns)
        
        # Select features (excluding target)
        feature_columns = [col for col in df.columns if col != target_column]
        temporal_features = ['breach_year', 'breach_month', 'breach_quarter',
                            'breach_day_of_week', 'is_weekend']
        
        available_features = []
        for col in feature_columns:
            if (col.startswith('organization_type_') or
                col.startswith('breach_type_') or
                col in temporal_features):
                available_features.append(col)
        
        X = df[available_features]
        y = df[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Regression Data Preparation Complete:")
        print(f"  Training samples: {X_train.shape[0]:,}")
        print(f"  Test samples: {X_test.shape[0]:,}")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Target mean: {y_train.mean():,.0f}")
        print(f"  Target median: {y_train.median():,.0f}")
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names after engineering.
        
        Returns:
            List of feature column names
        """
        return self.feature_names
    
    def __repr__(self):
        """String representation."""
        return f"FeatureEngineer(features={len(self.feature_names)})"