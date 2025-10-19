"""
Model Trainer Module
====================

Trains and optimizes machine learning models for breach prediction.
Implements multiple algorithms for comparison.

Classes:
    ModelTrainer: Main training orchestration class
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from typing import Dict, Any, Optional, Tuple
import pickle
import json
from datetime import datetime


class ModelTrainer:
    """
    Train and optimize machine learning models for breach prediction.
    
    Supports both classification (severe vs non-severe) and regression
    (predicting number affected) tasks.
    
    Attributes:
        models: Dictionary of trained models
        best_model: Best performing model
        training_history: Record of training sessions
        
    Example:
        >>> trainer = ModelTrainer()
        >>> model = trainer.train_severity_classifier(X_train, y_train)
        >>> trainer.save_model(model, 'severity_classifier.pkl')
    """
    
    def __init__(self):
        """Initialize the model trainer."""
        self.models = {}
        self.best_model = None
        self.training_history = []
        
    def train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series,
                                  **kwargs) -> LogisticRegression:
        """
        Train logistic regression classifier.
        
        Baseline model for binary classification. Fast and interpretable.
        Good starting point based on Assignment 5 analysis.
        
        Args:
            X_train: Training features
            y_train: Training target
            **kwargs: Additional parameters for LogisticRegression
            
        Returns:
            Trained LogisticRegression model
        """
        print("\nTraining Logistic Regression...")
        
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            **kwargs
        )
        
        model.fit(X_train, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        print(f"  Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        
        self.models['logistic_regression'] = model
        return model
    
    def train_random_forest_classifier(self, X_train: pd.DataFrame, y_train: pd.Series,
                                       n_estimators: int = 100,
                                       max_depth: Optional[int] = None,
                                       **kwargs) -> RandomForestClassifier:
        """
        Train Random Forest classifier.
        
        Ensemble method that handles non-linear relationships well.
        Based on EDA showing non-linear patterns in breach data.
        
        Args:
            X_train: Training features
            y_train: Training target
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            **kwargs: Additional parameters
            
        Returns:
            Trained RandomForestClassifier model
        """
        print("\nTraining Random Forest Classifier...")
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
            **kwargs
        )
        
        model.fit(X_train, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        print(f"  Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n  Top 5 Important Features:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")
        
        self.models['random_forest'] = model
        return model
    
    def train_gradient_boosting_classifier(self, X_train: pd.DataFrame, y_train: pd.Series,
                                          n_estimators: int = 100,
                                          learning_rate: float = 0.1,
                                          **kwargs) -> GradientBoostingClassifier:
        """
        Train Gradient Boosting classifier.
        
        Advanced ensemble method often achieving best performance.
        
        Args:
            X_train: Training features
            y_train: Training target
            n_estimators: Number of boosting stages
            learning_rate: Learning rate shrinkage
            **kwargs: Additional parameters
            
        Returns:
            Trained GradientBoostingClassifier model
        """
        print("\nTraining Gradient Boosting Classifier...")
        
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42,
            **kwargs
        )
        
        model.fit(X_train, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        print(f"  Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        
        self.models['gradient_boosting'] = model
        return model
    
    def train_all_classifiers(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Train all classification models and compare performance.
        
        Trains:
        - Logistic Regression (baseline)
        - Random Forest
        - Gradient Boosting
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Dictionary of trained models
            
        Example:
            >>> trainer = ModelTrainer()
            >>> models = trainer.train_all_classifiers(X_train, y_train)
            >>> print(f"Trained {len(models)} models")
        """
        print("=" * 60)
        print("TRAINING ALL CLASSIFICATION MODELS")
        print("=" * 60)
        
        # Train all models
        try:
            self.train_logistic_regression(X_train, y_train)
        except Exception as e:
            print(f"Error training Logistic Regression: {e}")
        
        try:
            self.train_random_forest_classifier(X_train, y_train)
        except Exception as e:
            print(f"Error training Random Forest: {e}")
        
        try:
            self.train_gradient_boosting_classifier(X_train, y_train)
        except Exception as e:
            print(f"Error training Gradient Boosting: {e}")
        
        print("\n" + "=" * 60)
        print(f"TRAINING COMPLETE: {len(self.models)} models trained")
        print("=" * 60)
        
        return self.models
    
    def train_random_forest_regressor(self, X_train: pd.DataFrame, y_train: pd.Series,
                                     n_estimators: int = 100,
                                     **kwargs) -> RandomForestRegressor:
        """
        Train Random Forest regressor for impact prediction.
        
        Predicts continuous values (number of individuals affected).
        
        Args:
            X_train: Training features
            y_train: Training target
            n_estimators: Number of trees
            **kwargs: Additional parameters
            
        Returns:
            Trained RandomForestRegressor model
        """
        print("\nTraining Random Forest Regressor...")
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1,
            **kwargs
        )
        
        model.fit(X_train, y_train)
        
        # Cross-validation score (R²)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        print(f"  Cross-validation R²: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n  Top 5 Important Features:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")
        
        self.models['random_forest_regressor'] = model
        return model
    
    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                                model_type: str = 'random_forest') -> Any:
        """
        Optimize hyperparameters using grid search.
        
        Performs grid search with cross-validation to find best parameters.
        
        Args:
            X_train: Training features
            y_train: Training target
            model_type: Type of model to optimize
            
        Returns:
            Optimized model
        """
        print(f"\nOptimizing {model_type} hyperparameters...")
        
        if model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
            
        elif model_type == 'gradient_boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
            base_model = GradientBoostingClassifier(random_state=42)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"  Best parameters: {grid_search.best_params_}")
        print(f"  Best CV accuracy: {grid_search.best_score_:.3f}")
        
        self.best_model = grid_search.best_estimator_
        self.models[f'{model_type}_optimized'] = self.best_model
        
        return self.best_model
    
    def save_model(self, model: Any, filepath: str, 
                  metadata: Optional[Dict] = None):
        """
        Save trained model to disk.
        
        Saves model using pickle and metadata using JSON.
        
        Args:
            model: Trained model to save
            filepath: Path to save model file
            metadata: Optional metadata dictionary
        """
        # Save model
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"\nModel saved to: {filepath}")
        
        # Save metadata if provided
        if metadata:
            metadata_path = filepath.replace('.pkl', '_metadata.json')
            metadata['saved_at'] = datetime.now().isoformat()
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Metadata saved to: {metadata_path}")
    
    @staticmethod
    def load_model(filepath: str) -> Any:
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to model file
            
        Returns:
            Loaded model
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from: {filepath}")
        return model
    
    def get_model_summary(self) -> pd.DataFrame:
        """
        Get summary of all trained models.
        
        Returns:
            DataFrame with model information
        """
        summary = []
        for name, model in self.models.items():
            summary.append({
                'model_name': name,
                'model_type': type(model).__name__,
                'n_features': model.n_features_in_ if hasattr(model, 'n_features_in_') else 'N/A'
            })
        
        return pd.DataFrame(summary)
    
    def __repr__(self):
        """String representation."""
        return f"ModelTrainer(models_trained={len(self.models)})"