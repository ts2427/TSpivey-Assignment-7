"""
Model Evaluator Module
======================

Evaluates model performance with business-relevant metrics.

Classes:
    ModelEvaluator: Performance evaluation and reporting
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """
    Evaluate model performance with comprehensive metrics.
    
    Provides both statistical metrics and business-relevant interpretations.
    
    Attributes:
        evaluation_results: Dictionary storing evaluation metrics
        
    Example:
        >>> evaluator = ModelEvaluator()
        >>> metrics = evaluator.evaluate_classifier(model, X_test, y_test)
        >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self.evaluation_results = {}
        
    def evaluate_classifier(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                           model_name: str = 'model') -> Dict[str, float]:
        """
        Comprehensive evaluation for classification models.
        
        Calculates:
        - Accuracy: Overall correctness
        - Precision: Of predicted severe breaches, how many are actually severe
        - Recall: Of actual severe breaches, how many we correctly identified
        - F1: Harmonic mean of precision and recall
        - ROC-AUC: Area under ROC curve
        
        Args:
            model: Trained classifier
            X_test: Test features
            y_test: Test target
            model_name: Name for storing results
            
        Returns:
            Dictionary of evaluation metrics
            
        Business Interpretation:
            - High recall: Catch most severe breaches (minimize false negatives)
            - High precision: Avoid false alarms (minimize false positives)
            - Balance based on cost of false positives vs false negatives
        """
        print(f"\n{'='*60}")
        print(f"EVALUATING: {model_name}")
        print(f"{'='*60}")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        # Print metrics
        print(f"\nPerformance Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1 Score:  {metrics['f1_score']:.3f}")
        if 'roc_auc' in metrics:
            print(f"  ROC-AUC:   {metrics['roc_auc']:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:  {cm[0,0]:,}")
        print(f"  False Positives: {cm[0,1]:,}")
        print(f"  False Negatives: {cm[1,0]:,}")
        print(f"  True Positives:  {cm[1,1]:,}")
        
        # Business interpretation
        print(f"\nBusiness Interpretation:")
        if metrics['recall'] > 0.8:
            print(f"  ✓ High recall ({metrics['recall']:.1%}): Successfully catching most severe breaches")
        else:
            print(f"  ⚠ Low recall ({metrics['recall']:.1%}): May miss some severe breaches")
            
        if metrics['precision'] > 0.8:
            print(f"  ✓ High precision ({metrics['precision']:.1%}): Few false alarms")
        else:
            print(f"  ⚠ Low precision ({metrics['precision']:.1%}): Some over-prediction of severity")
        
        # Store results
        metrics['confusion_matrix'] = cm
        metrics['classification_report'] = classification_report(y_test, y_pred)
        self.evaluation_results[model_name] = metrics
        
        return metrics
    
    def evaluate_regressor(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                          model_name: str = 'model') -> Dict[str, float]:
        """
        Comprehensive evaluation for regression models.
        
        Calculates:
        - RMSE: Root Mean Squared Error (same units as target)
        - MAE: Mean Absolute Error (average prediction error)
        - R²: Proportion of variance explained
        - MAPE: Mean Absolute Percentage Error
        
        Args:
            model: Trained regressor
            X_test: Test features
            y_test: Test target
            model_name: Name for storing results
            
        Returns:
            Dictionary of evaluation metrics
            
        Business Interpretation:
            - RMSE: Average prediction error in number of individuals
            - R²: How well model explains variation in impact
            - MAE: Typical prediction error magnitude
        """
        print(f"\n{'='*60}")
        print(f"EVALUATING: {model_name}")
        print(f"{'='*60}")
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        metrics = {
            'rmse': np.sqrt(mse),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2_score': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Percentage
        }
        
        # Print metrics
        print(f"\nPerformance Metrics:")
        print(f"  RMSE:     {metrics['rmse']:,.0f} individuals")
        print(f"  MAE:      {metrics['mae']:,.0f} individuals")
        print(f"  R² Score: {metrics['r2_score']:.3f}")
        print(f"  MAPE:     {metrics['mape']:.1f}%")
        
        # Additional statistics
        residuals = y_test - y_pred
        print(f"\nResidual Analysis:")
        print(f"  Mean residual:   {residuals.mean():,.0f}")
        print(f"  Median residual: {residuals.median():,.0f}")
        print(f"  Std residual:    {residuals.std():,.0f}")
        
        # Business interpretation
        print(f"\nBusiness Interpretation:")
        if metrics['r2_score'] > 0.7:
            print(f"  ✓ Strong predictive power (R²={metrics['r2_score']:.2f})")
        elif metrics['r2_score'] > 0.5:
            print(f"  ○ Moderate predictive power (R²={metrics['r2_score']:.2f})")
        else:
            print(f"  ⚠ Limited predictive power (R²={metrics['r2_score']:.2f})")
            
        print(f"  Average prediction error: ±{metrics['mae']:,.0f} individuals")
        
        # Store results
        self.evaluation_results[model_name] = metrics
        
        return metrics
    
    def compare_models(self, models: Dict[str, Any], X_test: pd.DataFrame, 
                      y_test: pd.Series, task: str = 'classification') -> pd.DataFrame:
        """
        Compare multiple models side-by-side.
        
        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test target
            task: 'classification' or 'regression'
            
        Returns:
            DataFrame comparing model performance
            
        Example:
            >>> comparison = evaluator.compare_models(
            ...     {'rf': rf_model, 'gb': gb_model}, 
            ...     X_test, y_test
            ... )
            >>> print(comparison.sort_values('accuracy', ascending=False))
        """
        print(f"\n{'='*60}")
        print(f"COMPARING {len(models)} MODELS")
        print(f"{'='*60}")
        
        results = []
        
        for name, model in models.items():
            if task == 'classification':
                metrics = self.evaluate_classifier(model, X_test, y_test, name)
                results.append({
                    'model': name,
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1_score'],
                    'roc_auc': metrics.get('roc_auc', np.nan)
                })
            else:  # regression
                metrics = self.evaluate_regressor(model, X_test, y_test, name)
                results.append({
                    'model': name,
                    'rmse': metrics['rmse'],
                    'mae': metrics['mae'],
                    'r2_score': metrics['r2_score']
                })
        
        comparison_df = pd.DataFrame(results)
        
        print(f"\n{'='*60}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(comparison_df.to_string(index=False))
        
        # Identify best model
        if task == 'classification':
            best_model = comparison_df.loc[comparison_df['f1_score'].idxmax(), 'model']
            print(f"\n🏆 Best model (by F1): {best_model}")
        else:
            best_model = comparison_df.loc[comparison_df['r2_score'].idxmax(), 'model']
            print(f"\n🏆 Best model (by R²): {best_model}")
        
        return comparison_df
    
    def plot_confusion_matrix(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                             model_name: str = 'Model', save_path: Optional[str] = None):
        """
        Plot confusion matrix heatmap.
        
        Args:
            model: Trained classifier
            X_test: Test features
            y_test: Test target
            model_name: Model name for title
            save_path: Optional path to save figure
        """
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Non-Severe', 'Severe'],
                   yticklabels=['Non-Severe', 'Severe'])
        plt.title(f'Confusion Matrix: {model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('Actual', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, model: Any, feature_names: list,
                               top_n: int = 15, save_path: Optional[str] = None):
        """
        Plot feature importance for tree-based models.
        
        Args:
            model: Trained model with feature_importances_
            feature_names: List of feature names
            top_n: Number of top features to show
            save_path: Optional path to save figure
        """
        if not hasattr(model, 'feature_importances_'):
            print("Model does not have feature_importances_ attribute")
            return
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Importance', fontsize=12)
        plt.title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to: {save_path}")
        
        plt.show()
    
    def generate_evaluation_report(self, model_name: str) -> str:
        """
        Generate text report of model evaluation.
        
        Args:
            model_name: Name of model to report on
            
        Returns:
            Formatted evaluation report
        """
        if model_name not in self.evaluation_results:
            return f"No evaluation results found for {model_name}"
        
        metrics = self.evaluation_results[model_name]
        
        report = f"""
MODEL EVALUATION REPORT: {model_name}
{'='*60}

PERFORMANCE METRICS:
"""
        
        if 'accuracy' in metrics:
            report += f"""
Classification Metrics:
  Accuracy:  {metrics['accuracy']:.3f}
  Precision: {metrics['precision']:.3f}
  Recall:    {metrics['recall']:.3f}
  F1 Score:  {metrics['f1_score']:.3f}
"""
            if 'roc_auc' in metrics:
                report += f"  ROC-AUC:   {metrics['roc_auc']:.3f}\n"
        
        if 'rmse' in metrics:
            report += f"""
Regression Metrics:
  RMSE:     {metrics['rmse']:,.0f}
  MAE:      {metrics['mae']:,.0f}
  R² Score: {metrics['r2_score']:.3f}
"""
        
        return report
    
    def __repr__(self):
        """String representation."""
        return f"ModelEvaluator(evaluated_models={len(self.evaluation_results)})"
