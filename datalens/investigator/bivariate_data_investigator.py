from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from scipy.stats import chi2_contingency, pearsonr, spearmanr

from ..visualization.visualizer import Visualizer
from .abstract_data_investigator import AbstractDataInvestigator


class BivariateDataInvestigator(AbstractDataInvestigator):
    """Perform Bivariate data analysis."""

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
        moderate_threshold: float,
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
        second_column: str,
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
            msg = f"Removed {original_count - complete_count} rows with missing values for correlation analysis"
            logging.info(msg)

        if complete_count < 2:  # noqa: PLR2004
            logging.warning("Insufficient data for correlation analysis after removing missing values")
            return

        try:
            # Calculate correlations
            pearson_stat, pearson_p_value = pearsonr(complete_data[first_column], complete_data[second_column])
            spearman_stat, spearman_p_value = spearmanr(complete_data[first_column], complete_data[second_column])

            # Create results DataFrame
            correlation_results = pd.DataFrame({
                "Correlation Type": ["Pearson", "Spearman"],
                "Coefficient": [pearson_stat, spearman_stat],
                "p-value": [pearson_p_value, spearman_p_value],
                "Significance": [pearson_p_value < SIGNIFICANCE_THRESHOLD,
                                spearman_p_value < SIGNIFICANCE_THRESHOLD],
                "Interpretation": [
                    self._interpret_correlation_strength(pearson_stat, STRONG_CORRELATION_THRESHOLD,
                                                        MODERATE_CORRELATION_THRESHOLD),
                    self._interpret_correlation_strength(spearman_stat, STRONG_CORRELATION_THRESHOLD,
                                                    MODERATE_CORRELATION_THRESHOLD),
                ],
            })

            # Display correlation results
            display(correlation_results)

            # Create visualizations
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Use the Visualizer methods
            self.visualizer.plot_regression(data, first_column, second_column, ax1)
            self.visualizer.plot_hexbin(data, first_column, second_column, ax2)

            plt.suptitle(
                f"Correlation Analysis\n"
                f"Pearson: r={pearson_stat:.3f} (p={pearson_p_value:.3f})\n"
                f"Spearman: r={spearman_stat:.3f} (p={spearman_p_value:.3f})",
                fontsize=14,
            )

            plt.tight_layout()
            plt.show()

        except Exception as e:
            log_msg = f"Error in categorical analysis: {e!s}"
            err_msg = f"Unable to complete categorical analysis: {e!s}"
            logging.exception(log_msg)
            raise ValueError(err_msg) from e

    def _analyze_categorical_relationship(
        self,
        data: pd.DataFrame,
        first_column: str,
        second_column: str,
    ) -> None:
        """Analyze the relationship between two categorical variables.

        This method examies the association between two categorical columns by:
        1. Creating contigency tables (raw counts and normalized)
        2. Performing chi-Square test of independence
        3. Visualizing the relationship with the stacked bar charts

        Args:
            data (pd.DataFrame): The dataset containing the columns to analyze
            first_column (str): Name of the first categorical column
            second_column (str): Name of the second categorical column

        """
        # Define constants
        SIGNIFICANCE_THRESHOLD = 0.05

        # Create contingency table
        contingency_table = pd.crosstab(data[first_column], data[second_column])


        try:
            # Calculate Chi-square test
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)

            # Check if test is reliable (expected frequencies should be >= 5)
            has_small_expected = (expected < 5).any()  # noqa: PLR2004

            # Create results DataFrame
            chi_square_results = pd.DataFrame({
                "Metric": ["Chi-square Statistic", "p-value", "Degrees of Freedom"],
                "Value": [f"{chi2:.3f}", f"{p_value:.3f}", dof],
            })

            # Display chi-square results and contingency table
            print(f"Association Analysis: {first_column} vs {second_column}")
            display(chi_square_results)

             # Display interpretation with warning if needed
            if has_small_expected:
                logging.warning("Some expected frequencies are < 5, chi-square may not be reliable")
                print("⚠️ Warning: Some expected frequencies are < 5, chi-square may not be reliable")

            # Show association strength interpretation
            significance = "significant" if p_value < SIGNIFICANCE_THRESHOLD else "not significant"
            print(f"The association between {first_column} and {second_column} is statistically {significance} (p={p_value:.3f})")  # noqa: E501

            # Display the contingency tables
            print("Raw Counts: Contingency Table")
            display(contingency_table)

            # Create and display normalized contingency table
            normalized_table = pd.crosstab(
                data[first_column],
                data[second_column],
                normalize="index",
                margins=True,
            )

            print("Normalized Proportions by Row")
            display(normalized_table)

            # Create visualizations
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Use the visualizer method for the plots
            self.visualizer.plot_stacked_bar(
                contingency_table,
                first_column,
                second_column,
                f"Distribution of {second_column} by {first_column}",
                ax1,
            )
            self.visualizer.plot_stacked_bar(
                normalized_table,
                first_column,
                second_column,
                f"Proportion of {second_column} by {first_column}",
                ax2,
            )

            plt.suptitle(
                f"Association Analysis: Chi-square={chi2:.2f}, p={p_value:.3f} ({significance})",
                fontsize=14,
            )

            plt.tight_layout()
            plt.show()
        except Exception as e:
            log_msg = f"Error in categorical analysis: {e!s}"
            err_msg = f"Unable to complete categorical analysis: {e!s}"
            logging.exception(log_msg)
            raise ValueError(err_msg) from e


    def _analyze_numerical_categorical_relationship(
        self,
        data: pd.DataFrame,
        numerical_column: str,
        categorical_column: str,
    ) -> None:
        """Analyze the relationship between numerical and categorical columns.

        Args:
            data (pd.DataFrame): The dataset containing the columns to analyze
            numerical_column (str): Name of the numerical column
            categorical_column (str): Name of the categorical column

        """
        try:
            # Calculate grouped statistic
            grouped_stats = (
                data
                .groupby(categorical_column)
                .agg(
                    count=(numerical_column, "count"),
                    missing=(numerical_column, lambda x: x.isna().sum()),
                    zeros=(numerical_column, lambda x: (x == 0).sum()),
                    min=(numerical_column, "min"),
                    q25=(numerical_column, lambda x: x.quantile(0.25)),
                    median=(numerical_column, "median"),
                    mean=(numerical_column, "mean"),
                    q75=(numerical_column, lambda x: x.quantile(0.75)),
                    max=(numerical_column, "max"),
                    std=(numerical_column, "std"),
                    iqr=(numerical_column, lambda x: x.quantile(0.75) - x.quantile(0.25)),
                )
            )

            # Add percentage of total for each category
            category_counts  = data[categorical_column].value_counts()
            category_percentages = category_counts /category_counts.sum() * 100

            # Create a DataFrame for displaying category distribution
            category_distribution = pd.DataFrame({
                "Count": category_counts,
                "Percentage": category_percentages.round(2),
            })

            # Display results in a structured manner
            print(f"Distribution of {numerical_column} by {categorical_column}")
            display(grouped_stats)

            print(f"Frequency of values in {categorical_column}")
            display(category_distribution)

            # Check if there are too many categories
            if data[categorical_column].nunique() > 10:  # noqa: PLR2004
                logging.warning("Large number of categories may make visualization cluttered")
                print("⚠️ Note: Only showing top 10 categories by frequency for visualization clarity.")
                top_categories = category_counts.nlargest(10).index.tolist()
                plot_data = data[data[categorical_column].isin(top_categories)].copy()
            else:
                plot_data = data.copy()

            # Create visualizations
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Use the Visualizer class for plotting
            self.visualizer.plot_violin_distribution(
                plot_data,
                categorical_column,
                numerical_column,
                ax1,
            )

            self.visualizer.plot_box_distribution(
                plot_data,
                categorical_column,
                numerical_column,
                ax2,
            )

            plt.suptitle(
                f"Distribution of {numerical_column} across {categorical_column} categories",
                fontsize=14,
            )

            plt.tight_layout()
            plt.show()

        except Exception as e:
            log_msg = f"Error in numerical-categorical analysis: {e!s}"
            err_msg = f"Unable to complete analysis: {e!s}"
            logging.exception(log_msg)
            raise ValueError(err_msg) from e

    def analyze_bivariate_relationship(
        self,
        first_column: str,
        second_column: str,
        data: pd.DataFrame | None = None,
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
        data = self.data.copy() if data is None else data

        # Validate columns exist using DataValidator
        self.validator.validate_column(first_column)
        self.validator.validate_column(second_column)

        # Determine column types
        is_first_numeric = pd.api.types.is_numeric_dtype(data[first_column])
        is_second_numeric = pd.api.types.is_numeric_dtype(data[second_column])

        if is_first_numeric and is_second_numeric:
            self._anlyze_numerical_relationship(data, first_column, second_column)
        elif not is_first_numeric and not is_second_numeric:
            self._analyze_categorical_relationship(data, first_column, second_column)
        else:
            numeric_column = first_column if is_first_numeric else second_column
            categorical_column = second_column if is_first_numeric else first_column
            self._analyze_numerical_categorical_relationship(data, numeric_column, categorical_column)
