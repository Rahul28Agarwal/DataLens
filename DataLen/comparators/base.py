from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .comparison_result import ComparisonResult


class BaseComparator(ABC):
    """Base class for comparing DataFrames."""

    def __init__(
        self,
        first_data: pd.DataFrame,
        second_data: pd.DataFrame,
        compare_cols: list[str]|None = None,
        feature_mapping: dict[str, str]|None = None,
    ) -> None:
       """Intialize the BaseDataComparator.

       Args:
            first_data (pd.DataFrame): The first DataFrame to compare.
            second_data (pd.DataFrame): The second DataFrame to compare.
            compare_cols (list[str] | None, optional): A list of column names to compare between the DataFrames.
                                                      Defaults to None.
            feature_mapping (dict[str, str] | None, optional): A dictionary mapping column names in df1 to their
                corresponding. Defaults to None.

       """
       self._first_data = None
       self._second_data = None
       self.first_data = first_data
       self.second_data = second_data
       self.compare_cols = compare_cols or first_data.columns.tolist()
       self.feature_mapping = feature_mapping or {col: col for col in self.compare_cols}

    @property
    def first_data(self) -> pd.DataFrame:
        """Getter for first_data."""
        return self._first_data

    @first_data.setter
    def first_data(self, data: pd.DataFrame) -> None:
        """Setter for first_data. Validates the input DataFrame.

        Args:
            data (pd.DataFrame): The DataFrame to set as first_data.

        """
        self._validate_dataframe(data, "first_data")
        self._first_data = data

    @property
    def second_data(self) -> pd.DataFrame:
        """Getter for second_data."""
        return self._second_data

    @second_data.setter
    def second_data(self, data: pd.DataFrame) -> None:
        """Setter for second_data. Validates the input DataFrame.

        Args:
            data (pd.DataFrame): The DataFrame to set as first_data.

        """
        self._validate_dataframe(data, "second_data")
        self._second_data = data

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame, name: str) -> None:
        """Validates that the input is a non-empty pandas DataFrame.

        Args:
            df: The DataFrame to validate.
            name: The name of the DataFrame (for error messages).

        Raises:
            ValueError: If df is not a pandas DataFrame or is empty.

        """  # noqa: D401
        if not isinstance(df, pd.DataFrame):
            msg = f"{name} must be a pandas DataFrame"
            raise TypeError(msg)
        if df.empty:
            msg = f"{name} cannot be empty"
            raise ValueError(msg)

    @abstractmethod
    def compare(self) -> list[ComparisonResult]:
        """Abstract method to perform comparison."""
