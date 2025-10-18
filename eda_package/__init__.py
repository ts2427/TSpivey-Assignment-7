"""
EDA Package for Data Breach Analysis
=====================================

A comprehensive exploratory data analysis package for analyzing data breach patterns,
trends, and business implications.

Modules:
    - analyzer: Statistical analysis classes and methods
    - visualizer: Data visualization classes
    - data_loader: Database connection and data loading utilities
    - insights: Business insights generation

Author: T. Spivey
Course: BUS 761
Date: October 2025
"""

__version__ = '1.0.0'
__author__ = 'T. Spivey'

from .analyzer import BreachAnalyzer
from .visualizer import BreachVisualizer
from .data_loader import DataLoader

__all__ = ['BreachAnalyzer', 'BreachVisualizer', 'DataLoader']