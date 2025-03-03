from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import seaborn as sns

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes

class PlotTheme:
    """Manages visual styling for plots.

    This class provides a centralized way to define and apply visual styles
    to matplotlib and seaborn plots. It encapsulates styling parameters
    and provides methods to apply them consistently across visualizations.

    Attributes:
        style (str): The seaborn style name (e.g., "whitegrid", "darkgrid").
        context (str): The plotting context (e.g., "notebook", "talk").
        palette (str): Color palette name.
        font_scale (float): Scaling factor for font sizes.

    """

    def __init__(
        self,
        style: str|None = "whitegrid",
        context: str|None = "notebook",
        palette: str|None = "deep",
        font_scale: float|None = 1.0,
    ) -> None:
        """Initialize a plot theme with styling parameters.

        Args:
            style: The seaborn style name. Options include "whitegrid", 
                "darkgrid", "white", "dark", and "ticks".
            context: The plotting context, which affects label sizes and 
                line weights. Options include "paper", "notebook", "talk", and "poster".
            palette: Color palette name. See seaborn's documentation for options.
            font_scale: Scaling factor for font sizes.
            figsize: Default figure size (width, height) in inches.

        """
        self.style = style
        self.context = context
        self.palette = palette
        self.font_scale = font_scale

    def apply(self) -> None:
        """Apply this theme to matplotlib/seaborn."""
        sns.set_theme(
            style=self.style,
            context=self.context,
            palette=self.palette,
            font_scale=self.font_scale,
        )

        # Set default figure size
        plt.rcParams["figure.figsize"] = self.figsize

class BasePlot(ABC):
    """Abstract base class for all plotting classes.

    This class defines the common interface and functionality for all plot types.
    Specific plot implementations should inherit from this class and implement
    the required methods.

    Attributes:
        theme (PlotTheme): The visual theme to apply to plots.

    """

    def __init__(self, theme: PlotTheme|None = None) -> None:
        """Initialize a base plot with an optional theme.

        Args:
            theme: The visual theme to apply to plots. If None, a default theme
                will be created.

        """
        self.theme = theme or PlotTheme()

    def apply_theme(self) -> None:
        """Apply the current theme."""
        self.theme.apply()

    @abstractmethod
    def create(self, data: pd.DataFrame, **kwargs: dict) -> Axes:
        """Create the plot visualization.

        This abstract method must be implemented by all concrete plot classes.

        Args:
            data: The pandas DataFrame containing the data to visualize.
            **kwargs: Additional keyword arguments specific to each plot type.

        Returns:
            The matplotlib Axes object containing the created plot.

        """

    def finalize_plot(  # noqa: PLR0913
        self,
        ax: Axes,
        title: str|None = None,
        xlabel: str|None = None,
        ylabel: str|None =None,
        legend: str|None =True,
        xtick_rotation: int|None = None,
    ) -> Axes:
        """Apply common formatting to finalize a plot.

        This method handles common plot finishing tasks like setting titles,
        labels, and legend visibility.

        Args:
            ax: The matplotlib Axes object to format.
            title: Plot title text. If None, no title will be set.
            xlabel: X-axis label text. If None, no label will be set.
            ylabel: Y-axis label text. If None, no label will be set.
            legend: Whether to show the legend if legend handles exist.
            xtick_rotation: Degrees of rotation for x-tick labels.
                If None, no rotation is applied.

        Returns:
            The formatted matplotlib Axes object.

        """
        # Set title if provided
        if title:
            ax.set_title(title)

        # Set axis labels if provided
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        # Show legend if requested and if legend handles exist
        if legend and ax.get_legend_handles_labels()[0]:
            ax.legend()

        # Rotate x-tick labels if specified
        if xtick_rotation is not None:
            plt.setp(ax.get_xticklabels(), rotation=xtick_rotation)

        return ax
