from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

from ..visualization.visualizer import Visualizer
from .abstract_data_investigator import AbstractDataInvestigator


class NumericalDataInvestigator(AbstractDataInvestigator):
    """Perform analysis on numerical variables in a DataFrame.

    This class extends AbstractDataInvestigator to provide specialized
    methods for investigating and visualizing numerical data. It offers
    tools for descriptive statistics and univariate analysis.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """Initialize the NumericalDataInvestigator.

        Args:
            data (pd.DataFrame): Pandas DataFrame.

        """
        super().__init__(data)
        self.visualizer = Visualizer()

    def describe_columns(self, data: pd.DataFrame | None = None) -> pd.DataFrame:
        """Calculate descriptive statistics for numerical columns in the dataset.

        This method analyzes the provided DataFrame (or the class's internal data if no DataFrame is provided)
        and focuses on columns with numerical data types. It calculates various statistics for each
        numerical column, including:

        * Count: Total number of non-null values
        * Missing: Number of missing values
        * Zeros: Number of zeros
        * Unique: Number of distinct values
        * Min, Max: Minimum and Maximum values
        * Quantiles: Percentiles (25th, 75th, 90th, 99th)
        * Mean, Median: Central tendency measures
        * StdDev: Standard deviation
        * Skew: Measure of asymmetry in the distribution
        * Kurtosis: Measure of peakedness in the distribution
        * IQR: Interquartile Range (difference between 75th and 25th percentile)

        Args:
            data (pd.DataFrame, optional): The DataFrame to analyze. Defaults to None (uses self.data).

        Returns:
            pd.DataFrame: A DataFrame containing the calculated descriptive statistics for each numerical column.

        """
        data = self.data.copy() if data is None else data

        numeric_data = data.select_dtypes(include="number")

        if numeric_data.empty:
            msg = "DataFrame has no numerical data."
            raise ValueError(msg)

        def calculate_stats(series: pd.Series) -> pd.Series:
            # Calculate a comprehensive set of statistics for the given series
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
        return stats_df[column_order].sort_values("Column")

    def univariate_analysis(
        self,
        column: str,
        data: pd.DataFrame | None = None,
        figsize: tuple[int, int] = (20,6),
        bins: int | None = None,
        show_plots: bool = True,  # noqa: FBT001
    ) -> tuple[pd.DataFrame, plt.figure] | None :
        """Perform univariate analysis on a numerical column in the dataset.

        This method analyzes the provided column (or a specific column if specified) and creates visualizations
        to explore its distribution. It calculates descriptive statistics and creates visualizations such as:

        * Descriptive Statistics: Count, Mean, Standard Deviation, Minimum, Maximum, etc.
            (calculated by num_column_describe)
        * Histogram: Shows the frequency distribution of the values
        * Box Plot: Displays the distribution with quartiles and potential outliers
        * Density Plot: Represents the probability density of the values

        Args:
            column (str): The name of the numerical column to analyze.
            data (pd.DataFrame, optional): The DataFrame containing the data. Defaults to None (uses self.data).
            figsize (tuple[int, int], optional): The figure size for the plots. Defaults to (20, 6).
            bins (int, optional): The number of bins for the histogram.  Defaults to None.
            show_plots (bool, optional): If True, displays the plots immediately. If False, returns
                the figure for further customization. Defaults to True.

        Returns:
            Optional[tuple[pd.DataFrame, plt.Figure]]: If show_plots is False, returns a tuple containing
            the descriptive statistics DataFrame and the matplotlib Figure object. If show_plots is True,
            returns None after displaying the plots.

        """
        data = self.data.copy() if data is None else data
        numeric_columns = data.select_dtypes(include="number").columns.tolist()

        if column not in numeric_columns:
            msg = f"Column {column} is not numerical or not present in the data."
            raise ValueError(msg)

        stats = self.describe_columns(data[[column]])
        display(stats.T)

        fig, axs = plt.subplots(1, 3, figsize=figsize)
        self.visualizer.plot_histogram(data, column, axs[0], bins)
        self.visualizer.plot_box(data, column, axs[1])
        self.visualizer.plot_ecdf(data, column, axs[2])

        plt.tight_layout()
        if show_plots:
            plt.show()
            return None
        return stats, fig



