from __future__ import annotations

import numpy as np
import pandas as pd
from IPython.display import display
import logging
from scipy.stats import chi2_contingency, pearsonr, spearmanr
from .abstract_data_investigator import AbstractDataInvestigator
from ..core.visualizer import Visualizer

class BivariateDataInvestigator(AbstractDataInvestigator):
    """Perform Bivariate data analysis"""

    def __init__(self, data: pd.DataFrame) -> None:
        """Initialize the BivariateDataInvestigator.

        Args:
            data (pd.DataFrame): Pandas DataFrame.
        """
        super().__init__(data)
        self.visualizer = Visualizer()
        
    def _interpret_correlation_strength(
        self,
        correlation_value: float, 
        strong_threshold: float, 
        moderate_threshold: float
    ) -> str:
        """Interpret the strength of a correlation coefficient.
        
        Args:
            correlation_value (float): The correlation coefficient
            strong_threshold (float): Threshold for strong correlation
            moderate_threshold (float): Threshold for moderate correlation
            
        Returns:
            str: Interpretation of correlation strength
        """
        correlation_abs = abs(correlation_value)
        direction = "positive" if correlation_value > 0 else "negative"
        
        if correlation_abs >= strong_threshold:
            strength = "strong"
        elif correlation_abs >= moderate_threshold:
            strength = "moderate"
        else:
            strength = "weak"
            
        return f"{strength} {direction}"
        
    def _anlyze_numerical_relationship(
        self,
        data: pd.DataFrame,
        first_column: str,
        second_column: str
    ) -> None:
        """Analyze the relationship between two numerical variables.
        This method performs comprehensive correlation analysis between two numerical columns,
        including Pearson (linear) and Spearman (monotonic) correlation calculations.
        It presents statistical interpretations and visualizations to help understand
        the strength, direction, and significance of the relationship.

        Args:
            data (pd.DataFrame): The dataset containing the columns to analyze
            first_column (str): Name of the first numerical column
            second_column (str): Name of the second numerical column
        """
        # Define interpretation thresholds as constants
        SIGNIFICANCE_THRESHOLD = 0.05
        STRONG_CORRELATION_THRESHOLD = 0.7
        MODERATE_CORRELATION_THRESHOLD = 0.3
        
        # Handle missing values
        complete_data = data[[first_column, second_column]].dropna()
        original_count = len(data)
        complete_count = len(complete_data)
        
        if complete_count < original_count:
            logging.info(f"Removed {original_count - complete_count} rows with missing values for correlation analysis")
        
        if complete_count < 2:
            logging.warning("Insufficient data for correlation analysis after removing missing values")
            return
        
        try:
            # Calculate correlations
            pearson_stat, pearson_p_value = pearsonr(complete_data[first_column], complete_data[second_column])
            spearman_stat, spearman_p_value = spearmanr(complete_data[first_column], complete_data[second_column])
            
            # Create results DataFrame
            correlation_results = pd.DataFrame({
                'Correlation Type': ['Pearson', 'Spearman'],
                'Coefficient': [pearson_stat, spearman_stat],
                'p-value': [pearson_p_value, spearman_p_value],
                'Significance': [pearson_p_value < SIGNIFICANCE_THRESHOLD, 
                                spearman_p_value < SIGNIFICANCE_THRESHOLD],
                'Interpretation': [
                    self._interpret_correlation_strength(pearson_stat, STRONG_CORRELATION_THRESHOLD, 
                                                        MODERATE_CORRELATION_THRESHOLD),
                    self._interpret_correlation_strength(spearman_stat, STRONG_CORRELATION_THRESHOLD, 
                                                    MODERATE_CORRELATION_THRESHOLD)
                ]
            })
            
            # Display correlation results
            display(correlation_results)
            
        except Exception as e:
            raise e
        
    def analyze_bivariate_relationship(
        self,
        first_column: str,
        second_column: str, 
        data: pd.DataFrame | None = None
    ) -> None:
        """Perform comprehensive bivariate analysis between two columns.
        
         This method automatically detects the data types of the provided columns
        and dispatches the analysis to the appropriate specialized method:
        - Numeric vs. Numeric: Correlation analysis with scatterplots
        - Categorical vs. Categorical: Contingency tables and chi-square tests
        - Numeric vs. Categorical: Distribution comparison across categories
        
        Args:
            first_column (str): Name of the first column to analyze
            second_column (str): Name of the second column to analyze
            data (pd.DataFrame | None, optional): The dataset to analyze. 
                If None, uses the instance's data. Defaults to None.
        """
        # Use instance data if none provided
        data = data or self.data.copy()
        
        # Validate columns exist using DataValidator
        self.validator.validate_column(first_column)
        self.validator.validate_column(second_column)
        
        # Determine column types 
        is_first_numeric = pd.api.types.is_numeric_dtype(data[first_column])
        is_second_numeric = pd.api.types.is_numeric_dtype(data[second_column])
        
        if is_first_numeric and is_second_numeric:
            pass
        elif not is_first_numeric and not is_second_numeric:
            pass
        else:
            numeric_column = first_column if is_first_numeric else second_column
            categorical_column = second_column if is_first_numeric else first_column