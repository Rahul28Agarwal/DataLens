from __future__ import annotations

import pandas as pd
from .abstract_data_investigator import AbstractDataInvestigator
from ..core.visualizer import Visualizer

class NumericalDataInvestigator(AbstractDataInvestigator):
    """Perform analysis on numerical variables"""

    def __init__(self, data: pd.DataFrame) -> None:
        """Initialize the NumercialDataInvestigator.

        Args:
            data (pd.DataFrame): Pandas DataFrame.
        """
        super().__init__(data)
        self.visualizer = Visualizer()

    def describe_columns(self, data: pd.DataFrame | None = None) -> pd.DataFrame:
        data = data or self.data.copy()
        
        numeric_data = data.select_dtypes(include="number")
        
        if numeric_data.empty:
            raise ValueError("DataFrame has no numerical data.")

        def calculate_stats(series: pd.Series):
            return pd.Series({
                "Count": series.count(),
                "Missing": series.isna().sum(),
                "Zeros": (series == 0).sum(),
                "Unique": series.nunique(),
                "Min": series.min(),
                "25%": series.quantile(0.25),
                "Mean": series.mean(),
                "Median": series.median(),
                "75%": series.quantile(0.75),
                "90%": series.quantile(0.90),
                "99%": series.quantile(0.99),
                "Max": series.max(),
                "StdDev": series.std(),
                "Skew": series.skew(),
                "Kurtosis": series.kurtosis(),
                "IQR": series.quantile(0.75) - series.quantile(0.25),
            }, name=series.name)
        
        stats_df = numeric_data.apply(calculate_stats).T.reset_index().rename(columns={"index": "Column"})
        stats_df["Data Type"] = numeric_data.dtypes.values

        column_order = ["Column", "Data Type"] + [col for col in stats_df.columns if col not in {"Column", "Data Type"}]
        return stats_df[column_order].sort_values("column")

