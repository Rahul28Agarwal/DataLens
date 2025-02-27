from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if TYPE_CHECKING:
    import pandas as pd


class Visualizer:
    """A class for generating visualization for data analysis."""

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
        """Plot a bar plot.

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

    def plot_regression(
            self,
            data: pd.DataFrame,
            x_column: str,
            y_column: str,
            ax: plt.Axes,
            color: str = "blue",
            scatter_kws: dict| None = None,
            line_kws: dict| None = None,
        ) -> None:
        """Plot a scatter plot with regression line.

        Args:
            data (pd.DataFrame): Input pandas DataFrame
            x_column (str): Name of column for x-axis
            y_column (str): Name of column for y-axis
            ax (plt.Axes): Graph's axis object
            color (str, optional): Color for the plot. Defaults to "blue".
            scatter_kws (dict, optional): Additional keywords for scatter plot. Defaults to None.
            line_kws (dict, optional): Additional keywords for regression line. Defaults to None.

        """
        # Set default keyword arguments if None
        scatter_kws = scatter_kws or {"alpha": 0.6}
        line_kws = line_kws or {"color": color}

        sns.regplot(
            data=data,
            x=x_column,
            y=y_column,
            ax=ax,
            scatter_kws=scatter_kws,
            line_kws=line_kws,
        )
        ax.set_title(f"Relationship between {x_column} and {y_column}")
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)

    def plot_hexbin(
        self,
        data: pd.DataFrame,
        x_column: str,
        y_column: str,
        ax: plt.Axes,
        gridsize: int = 30,
        cmap: str = "Blues",
        show_colorbar: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        """Plot a hexbin density visualization.

        This plot is useful for visualizing the density of points when dealing with large datasets,
        showing where most data points are concentrated.

        Args:
            data (pd.DataFrame): Input pandas DataFrame
            x_column (str): Name of column for x-axis
            y_column (str): Name of column for y-axis
            ax (plt.Axes): Graph's axis object
            gridsize (int, optional): Size of the hexagonal bins. Defaults to 30.
            cmap (str, optional): Colormap for the hexbin. Defaults to "Blues".
            show_colorbar (bool, optional): Whether to show colorbar. Defaults to True.

        """
        # Create the hexbin plot
        hb = ax.hexbin(
            data[x_column],
            data[y_column],
            gridsize=gridsize,
            cmap=cmap,
        )

        ax.set_title(f"Density plot of {x_column} vs {y_column}")
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)

        if show_colorbar:
            plt.colorbar(hb, ax=ax, label="Count")

    def plot_stacked_bar(
        self,
        data: pd.DataFrame,
        x_column: str,
        y_column: str,
        title: str,  # noqa: ARG002
        ax: plt.Axes,
        colormap: str = "Accent",
    ) -> None:
        """Plot a stacked bar chart for categorical data comparison.

        Args:
            data (pd.DataFrame): Input contingency table
            x_column (str): Name of column for x-axis
            y_column (str): Name of column for y-axis
            title (str): Title for the plot
            ax (plt.Axes): Graph's axis object
            colormap (str, optional): Colormap for the bars. Defaults to "Accent".

        """
        data.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            colormap=colormap,
        )

        ax.set_title(f"Stacked Bar plot for {x_column} by {y_column}")
        ax.set_ylabel("Count")
        ax.set_xlabel(x_column)
        ax.tick_params(axis="x", rotation=45)

    def plot_violin_distribution(
        self,
        data: pd.DataFrame,
        category_column: str,
        value_column: str,
        ax: plt.Axes,
        palette: str = "muted",
    ) -> None:
        """Plot a violin plot showing distribution of numerical values across categories.

        Args:
            data (pd.DataFrame): Input pandas DataFrame
            category_column (str): Name of categorical column for x-axis
            value_column (str): Name of numerical column for y-axis
            ax (plt.Axes): Graph's axis object
            palette (str, optional): Color palette for the plot. Defaults to "muted".

        """
        # Create violin plot with interior points
        sns.violinplot(
            x=category_column,
            y=value_column,
            data=data,
            ax=ax,
            hue=value_column,
            palette=palette,
            inner="point",  # Show individual data points
        )

        # Add mean markers
        means = data.groupby(category_column)[value_column].mean()
        for i, mean_val in enumerate(means):
            ax.plot(i, mean_val, "o", color="red", markersize=8)

        ax.set_title(f"Distribution of {value_column} by {category_column}")
        ax.set_xlabel(category_column)
        ax.set_ylabel(value_column)
        ax.tick_params(axis="x", rotation=45)

    def plot_box_distribution(
        self,
        data: pd.DataFrame,
        category_column: str,
        value_column: str,
        ax: plt.Axes,
        palette: str = "muted",
    ) -> None:
        """Plot a box plot showing distribution of numerical values across categories.

        Args:
            data (pd.DataFrame): Input pandas DataFrame
            category_column (str): Name of categorical column for x-axis
            value_column (str): Name of numerical column for y-axis
            ax (plt.Axes): Graph's axis object
            palette (str, optional): Color palette for the plot. Defaults to "muted".

        """
        # Create box plot with stripped points
        sns.boxplot(
            x=category_column,
            y=value_column,
            data=data,
            ax=ax,
            hue=value_column,
            palette=palette,
            showfliers=False,  # Hide outliers as they'll be shown in strip plot
        )

        # Add strip plot for individual points
        sns.stripplot(
            x=category_column,
            y=value_column,
            data=data,
            ax=ax,
            color="black",
            alpha=0.3,
            size=3,
            jitter=True,
        )

        # Add category size annotations
        sizes = data[category_column].value_counts()
        for i, (_, size) in enumerate(sizes.items()):
            ax.annotate(
                f"n={size}",
                xy=(i, data[value_column].min()),
                xytext=(0, -20),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=8,
            )

        ax.set_title(f"Box Plot of {value_column} by {category_column}")
        ax.set_xlabel(category_column)
        ax.set_ylabel(value_column)
        ax.tick_params(axis="x", rotation=45)
