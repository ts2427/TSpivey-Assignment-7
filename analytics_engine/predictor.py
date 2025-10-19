"""
Breach Predictor Module
========================

Production-ready prediction interface for trained models.

Classes:
    BreachPredictor: Main prediction interface
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List
import pickle


class BreachPredictor:
    """
    Production prediction interface for breach severity and impact models.
    
    Loads trained models and provides simple prediction API.
    
    Attributes:
        model: Loaded trained model
        model_type: Type of model (classifier or regressor)
        feature_names: Expected feature names
        
    Example:
        >>> predictor = BreachPredictor()
        >>> predictor.load_model('models/severity_classifier.pkl')
        >>> risk_score = predictor.predict_severity(breach_data)
        >>> print(f"Severity risk: {risk_score:.1%}")
    """
    
    def __init__(self):
        """Initialize the predictor."""
        self.model = None
        self.model_type = None
        self.feature_names = []
        
    def load_model(self, model_path: str):
        """
        Load a trained model from file.
        
        Args:
            model_path: Path to pickled model file
        """
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Determine model type
        model_class = type(self.model).__name__
        if 'Regressor' in model_class or 'Ridge' in model_class or 'Lasso' in model_class:
            self.model_type = 'regressor'
        else:
            self.model_type = 'classifier'
        
        # Get feature names if available
        if hasattr(self.model, 'feature_names_in_'):
            self.feature_names = self.model.feature_names_in_.tolist()
        
        print(f"Model loaded: {model_class} ({self.model_type})")
        if self.feature_names:
            print(f"Expected features: {len(self.feature_names)}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities (classifiers only).
        
        Args:
            X: Feature dataframe
            
        Returns:
            Array of class probabilities
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        if self.model_type != 'classifier':
            raise ValueError("predict_proba only available for classifiers")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model does not support probability prediction")
        
        return self.model.predict_proba(X)
    
    def predict_severity(self, X: pd.DataFrame) -> Union[np.ndarray, float]:
        """
        Predict breach severity (returns probability for severe class).
        
        For business use: Returns risk score between 0-1.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Severity probability/probabilities
            
        Business Interpretation:
            - 0.0-0.3: Low severity risk
            - 0.3-0.7: Medium severity risk
            - 0.7-1.0: High severity risk
        """
        if self.model_type != 'classifier':
            raise ValueError("predict_severity only available for classifiers")
        
        proba = self.predict_proba(X)
        
        # Return probability of severe class (class 1)
        if len(proba.shape) > 1:
            return proba[:, 1]
        else:
            return proba[1]
    
    def predict_impact(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict breach impact (number of individuals affected).
        
        Args:
            X: Feature dataframe
            
        Returns:
            Predicted impact values
        """
        if self.model_type != 'regressor':
            raise ValueError("predict_impact only available for regressors")
        
        return self.predict(X)
    
    def predict_single(self, features: Dict[str, Any]) -> Union[float, int]:
        """
        Make prediction for a single breach scenario.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Single prediction value
            
        Example:
            >>> prediction = predictor.predict_single({
            ...     'organization_type': 'MED',
            ...     'breach_type': 'HACK',
            ...     'breach_year': 2025
            ... })
        """
        # Convert to dataframe
        X = pd.DataFrame([features])
        
        # Make prediction
        if self.model_type == 'classifier':
            return self.predict_severity(X)[0]
        else:
            return self.predict_impact(X)[0]
    
    def batch_predict(self, data: pd.DataFrame, 
                     return_risk_level: bool = True) -> pd.DataFrame:
        """
        Make predictions on batch of breach scenarios.
        
        Args:
            data: Dataframe of breach features
            return_risk_level: Whether to add risk level labels
            
        Returns:
            Dataframe with predictions added
        """
        result = data.copy()
        
        if self.model_type == 'classifier':
            # Get probabilities
            result['severity_probability'] = self.predict_severity(data)
            result['predicted_severe'] = (result['severity_probability'] > 0.5).astype(int)
            
            if return_risk_level:
                result['risk_level'] = pd.cut(
                    result['severity_probability'],
                    bins=[0, 0.3, 0.7, 1.0],
                    labels=['Low', 'Medium', 'High']
                )
        else:
            # Regression predictions
            result['predicted_impact'] = self.predict_impact(data)
            
            if return_risk_level:
                # Categorize by impact magnitude
                result['impact_level'] = pd.cut(
                    result['predicted_impact'],
                    bins=[0, 1000, 10000, np.inf],
                    labels=['Small', 'Medium', 'Large']
                )
        
        return result
    
    def explain_prediction(self, X: pd.DataFrame, index: int = 0) -> Dict[str, Any]:
        """
        Provide explanation for a single prediction.
        
        Args:
            X: Feature dataframe
            index: Row index to explain
            
        Returns:
            Dictionary with prediction explanation
        """
        if self.model is None:
            raise ValueError("No model loaded")
        
        # Get prediction
        if self.model_type == 'classifier':
            prediction = self.predict_severity(X.iloc[[index]])[0]
            pred_class = 'Severe' if prediction > 0.5 else 'Non-Severe'
        else:
            prediction = self.predict_impact(X.iloc[[index]])[0]
            pred_class = None
        
        explanation = {
            'prediction': prediction,
            'predicted_class': pred_class,
            'features': X.iloc[index].to_dict()
        }
        
        # Add feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            explanation['top_features'] = feature_importance.head(5).to_dict('records')
        
        return explanation
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded model.
        
        Returns:
            Dictionary with model metadata
        """
        if self.model is None:
            return {'status': 'No model loaded'}
        
        info = {
            'model_class': type(self.model).__name__,
            'model_type': self.model_type,
            'n_features': len(self.feature_names) if self.feature_names else 'Unknown',
            'feature_names': self.feature_names
        }
        
        # Add model-specific info
        if hasattr(self.model, 'n_estimators'):
            info['n_estimators'] = self.model.n_estimators
        
        if hasattr(self.model, 'max_depth'):
            info['max_depth'] = self.model.max_depth
        
        return info
    
    def __repr__(self):
        """String representation."""
        if self.model is None:
            return "BreachPredictor(no model loaded)"
        return f"BreachPredictor(model={type(self.model).__name__}, type={self.model_type})"
