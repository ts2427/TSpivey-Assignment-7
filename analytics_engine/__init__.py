"""
Analytics Engine for Data Breach Prediction
============================================

Machine learning models for predicting breach severity and impact.
Builds on the exploratory data analysis from Assignment 5.

Modules:
    - feature_engineer: Data preprocessing and feature engineering
    - model_trainer: Model training and optimization
    - predictor: Prediction interface
    - evaluator: Model evaluation and metrics
    - recommender: Business recommendations based on predictions

Author: T. Spivey
Course: BUS 761
Date: October 2025
"""

__version__ = '1.0.0'
__author__ = 'T. Spivey'

from .feature_engineer import FeatureEngineer
from .model_trainer import ModelTrainer
from .predictor import BreachPredictor
from .evaluator import ModelEvaluator
from .recommender import BusinessRecommender

__all__ = [
    'FeatureEngineer',
    'ModelTrainer', 
    'BreachPredictor',
    'ModelEvaluator',
    'BusinessRecommender'
]
