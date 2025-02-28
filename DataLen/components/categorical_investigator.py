from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from ..core.visualizer import Visualizer
from .abstract_data_investigator import AbstractDataInvestigator


class CategoricalDataInvestigator(AbstractDataInvestigator):
    """Perform analysis on categorical variables."""

    def __init__(self, data: pd.DataFrame) -> None:
        """Initialize the CategoricalDataInvestigator.

        Args:
            data (pd.DataFrame): Pandas DataFrame.

        """
        super().__init__(data)
        self.visualizer = Visualizer()

    def describe_columns(self, data: pd.DataFrame | None = None) -> pd.DataFrame:
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

        """  # noqa: D401
        data = data or self.data.copy()

        # Get dataframe with only categorical columns
        categorical_data = data.select_dtypes(exclude="number")

        if categorical_data.empty:
            msg = "DataFrame has no categorical data."
            raise ValueError(msg)

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
            }, name=series.name)
        stats_df = categorical_data.apply(calculate_stats).T.reset_index().rename(columns={"index":"Column"})
        stats_df["Data Type"] = categorical_data.dtypes.values

        column_order = ["Column", "Data Type"] + [col for col in stats_df.columns if col not in {"Column", "Data Type"}]
        return stats_df[column_order].sort_values("Column")

    @staticmethod
    def _calculate_entropy(series: pd.Series) -> float:
        value_counts = series.value_counts(normalize=True)
        return -np.sum(value_counts * np.log2(value_counts))

    def univariate_analysis(
            self,
            column: str,
            data: pd.DataFrame | None = None,
            figsize: tuple[int, int] = (20,6),
            show_plots: bool = True,  # noqa: FBT001, FBT002
    ) -> tuple[pd.DataFrame, plt.figure] | None:
        """Perform univariate analysis on a categorical column in the dataset.

        Args:
            column (str): The name of the categorical column to analyze.
            data (pd.DataFrame | None, optional): The DataFrame containing the data.. Defaults to None.
            figsize (tuple[int, int], optional): The figure size for the plots. Defaults to (20,6).
            show_plots (bool, optional): If True, displays the plots immediately. If False, returns
                the figure for further customization. Defaults to True.

        Returns:
            Optional[tuple[pd.DataFrame, plt.figure]]: If show_plots is False, returns a tuple containing
            the descriptive statistics DataFrame and the matplotlib Figure object. If show_plots is True,
            returns None after displaying the plots.

        """
        data = data or self.data.copy()
        categorical_columns = data.select_dtypes(exclude="number").columns.tolist()

        if column not in categorical_columns:
            msg = f"Column {column} is not numerical or not present in the data."
            raise ValueError(msg)

        stats = self.describe_columns(data[[column]])
        print(f"Descriptive statistics for {column}")
        display(stats)

        unique_values = data[column].nunique()
        top_n = min(unique_values, 10)

        frequency = self.data[column].value_counts().reset_index()
        frequency.columns = [column, "Count"]
        frequency["Percentage"] = (frequency["Count"] / frequency["Count"].sum()) * 100

        print(f"\nTop {top_n} frequency counts for {column}")
        display(frequency)

        fig, axs = plt.subplots(1, 2, figsize=figsize)
        self.visualizer.plot_bar(data, x_column=column, y_column="Count", ax=axs[0])
        self.visualizer.plot_pie(data, x_column=column, y_column="Count", ax=axs[1])

        plt.tight_layout()
        if show_plots:
            plt.show()
            return None
        return stats, fig



