from __future__ import annotations
from typing import Optional
import matplotlib.pyplot as plt
from IPython.display import display

import pandas as pd
import numpy as np
from .abstract_data_investigator import AbstractDataInvestigator
from ..core.visualizer import Visualizer

class CategoricalDataInvestigator(AbstractDataInvestigator):
    """Perform analysis on categorical variables"""

    def __init__(self, data: pd.DataFrame) -> None:
        """Initialize the CategoricalDataInvestigator

        Args:
            data (pd.DataFrame): Pandas DataFrame.
        """
        super().__init__(data)
        self.visualizer = Visualizer()

    def descibe_column(self, data: pd.DataFrame | None = None) -> pd.DataFrame:
        """Calculates descriptive statistics for categorical columns in the dataset.

        This method analyzes the provided DataFrame (or the class's internal data if no DataFrame is provided)
        and focuses on columns with categorical data types. It calculates various statistics for each
        categorical column, including:

        * Count: Total number of non-null values
        * Missing: Number of missing values
        * Unique: Number of distinct categories
        * Mode: Most frequent category value
        * Mode Frequency: Count of the mode value
        * Mode Percentage: Percentage of observations belonging to the mode category
        * Second Mode: Second most frequent category value (if it exists)
        * Entropy: Measure of uncertainty or disorder in the categorical distribution
            (using private method _calculate_entropy)

        Args:
            data (pd.DataFrame, optional): The DataFrame to analyze. Defaults to None (uses self.data).

        Returns:
            pd.DataFrame: A DataFrame containing the calculated descriptive statistics for each categorical column.

        """ 
        data = data or self.data.copy()
        
        # Get dataframe with only categorical columns
        categorical_data = data.select_dtypes(exclude="number")

        if categorical_data.empty:
            raise ValueError("DataFrame has no categorical data.")
        
        def calculate_stats(series: pd.Series) -> pd.Series:
            return pd.Series({
                "Count": series.count(),
                "Missing": series.isna().sum(),
                "Unique": series.nunique(),
                "Mode": series.mode().iloc[0] if not series.mode().empty else None,
                "Mode Frequency": series.value_counts().iloc[0] if not series.value_counts().empty else 0,
                "Mode Percentage": (series.value_counts(normalize=True).iloc[0] * 100) if not series.value_counts().empty else 0,  # noqa: E501
                "Second Mode": series.mode().iloc[1] if len(series.mode()) > 1 else None,
                "Entropy": self._calculate_entropy(series),
            })
        stats_df = categorical_data.apply(calculate_stats).T.reset_index().rename(columns={"index":"Column"})
        stats_df["Data Type"] = categorical_data.dtypes.values

        column_order = ["column", "Data Type"] + [col for col in stats_df.columns if col not in {"Column", "Data Type"}]
        return stats_df[column_order].sort_values("Column")
    
    @staticmethod
    def _calculate_entropy(series: pd.Series) -> float:
        value_counts = series.value_counts(normalize=True)
        return -np.sum(value_counts * np.log2(value_counts))
