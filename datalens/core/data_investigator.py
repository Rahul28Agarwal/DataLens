from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from ..investigator.abstract_data_investigator import AbstractDataInvestigator
from ..investigator.bivariate_data_investigator import BivariateDataInvestigator
from ..investigator.categorical_investigator import CategoricalDataInvestigator
from ..investigator.numerical_investigator import NumericalDataInvestigator

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


class DataInvestigator(AbstractDataInvestigator):
    """A class for performing comprehensive data investigation.

    This class combines functionality from specialized investigators (numerical,
    categorical, and bivariate) to provide automatic and thorough data analysis.
    It detects data types, provides summary statistics, visualizes distributions,
    and identifies potentially interesting relationships in the data.

    """

    def __init__(self, data: pd.DataFrame) -> None:
        """Initialize the DataInvestigator.

        Args:
            data (pd.DataFrame): DataFrame to be analyzed.
        """
        super().__init__(data)
        self.numerical_investigator = NumericalDataInvestigator(data)
        self.categorical_investigator = CategoricalDataInvestigator(data)
        self.bivariate_investigator = BivariateDataInvestigator(data)

    def profile_data(self, data: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate a comprehensive profile of the dataset.

        This method creates a high-level overview of the dataset including:
        - Basic dataframe info (shape, memory usage)
        - Column types and counts
        - Missing value analysis
        - Duplicated rows

        Args:
            data (pd.DataFrame | None, optional): DataFrame to analyze. Defaults to None.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: [describe the numerical columns, describe the categorical columns]

        """
        data = self.data.copy() if data is None else data

        # Describe the numerical columns
        numercial_describe = self.numerical_investigator.describe_columns(data)

        # Describe categorical columns
        categorical_describe = self.categorical_investigator.describe_columns(data)

        return numercial_describe, categorical_describe

    def univariate_analysis(
        self,
        column: str,
        data: pd.DataFrame | None = None,
        figsize: tuple[int, int] = (20,6),
        bins: int | None = None,
        show_plots: bool = True,  # noqa: FBT001
    ) -> tuple[pd.DataFrame, plt.figure] | None :
        """Perform univariate analysis on a column in the dataset.

        This method analyzes the provided column (or a specific column if specified) and creates visualizations
        to explore its distribution.

        Args:
            column (str): The name of the numerical column to analyze.
            data (pd.DataFrame, optional): The DataFrame containing the data. Defaults to None (uses self.data).
            figsize (tuple[int, int], optional): The figure size for the plots. Defaults to (20, 6).
            bins (int, optional): The number of bins for the histogram.  Defaults to None.
                Only for the numerical columns.
            show_plots (bool, optional): If True, displays the plots immediately. If False, returns
                the figure for further customization. Defaults to True.

        Returns:
            Optional[tuple[pd.DataFrame, plt.Figure]]: If show_plots is False, returns a tuple containing
            the descriptive statistics DataFrame and the matplotlib Figure object. If show_plots is True,
            returns None after displaying the plots.

        """
        data = data or self.data.copy()
        if pd.api.types.is_numeric_dtype(data[column]):
            stats, fig = self.numerical_investigator.univariate_analysis(column, data, figsize, bins, show_plots)
        else:
            stats, fig = self.categorical_investigator.univariate_analysis(column, data, figsize, show_plots)

        return stats, fig

    def bivariate_analysis(
        self,
        first_column: str,
        second_column: str,
        data: pd.DataFrame | None = None,
    )-> None:
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
        return self.bivariate_investigator.analyze_bivariate_relationship(first_column, second_column, data)

