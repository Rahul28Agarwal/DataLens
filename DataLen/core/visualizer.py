from  __future__ import annotations
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class Visualizer:
    """A class for generating visualization for data analysis."""

    def __init__(self):
        pass

    def plot_histogram(
            self,
            data: pd.DataFrame,
            column: str,
            ax: plt.axes, 
            bins: int | None = None,
    ) -> None:
        """Plot a histogram with automatic bin calculation.

        Args:
            data (pd.DataFrame): Input pandas DataFrame
            column (str): Name of column for histogram.
            ax (plt.Axes): Graphs axis object
            bins (int | None, optional): Number of bins for histogram. Defaults to None.
        """
        series = data[column].dropna()

        # Get the range of the columns
        data_range = series.max() - series.min()

        # Calculate the number of bins
        bins = bins or min(int(np.ceil(data_range/20)), 50)
        bins = max(bins, 10)

        sns.histplot(series, bins=bins, kde=True, ax=ax)
        ax.set_title(f"Histogram of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
    
    def plot_box(self, data: pd.DataFrame, column: str, ax: plt.axes) -> None:
        """Plot a box plot.

        Args:
            data (pd.DataFrame): Input Pandas DataFrame.
            column (str): Name of the column for box plot.
            ax (plt.axes): Graph's axis object
        """
        sns.boxplot(x=data[column], ax=ax)
        ax.set_title(f"Box plot of {column}")
    
    def plot_ecdf(self, data: pd.DataFrame, column: str, ax: plt.axes) -> None:
        """Plot an empirical cumulative distribution function.

        Args:
            data (pd.DataFrame): Input Pandas DataFrame.
            column (str): Name of the column for ecdf plot.
            ax (plt.axes): Graph's axis object
        """
        sns.ecdfplot(data=data, x=column, ax=ax)
        ax.set_title(f"ECDF of {column}")

    def plot_bar(self, data: pd.DataFrame, x_column: str, y_column: str, ax: plt.axes) -> None:
        """Plat a bar plot

        Args:
            data (pd.DataFrame): Input Pandas DataFrame.
            x_column (str): Name of the x axis column for bar plot.
            y_column (str): Name of the y axis column for bar plot.
            ax (plt.axes): Graph's axis object
        """
        sns.barplot(x=x_column, y=y_column, data=data, ax=ax)
        ax.set_title(f"Bar chart of {x_column} and {y_column}")
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.tick_params(axis="x", rotation=45)

    def plot_pie(self, data: pd.DataFrame, x_column: str, y_column: str, ax: plt.axes) -> None:
        ax.pie(data[y_column], labels=data[x_column], autopct="%1.1f%%", startangle=90)
        ax.set_title(f"Pie chart of {x_column}")