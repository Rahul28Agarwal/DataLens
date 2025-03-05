from __future__ import annotations  # noqa: D100

import numpy as np
import pandas as pd

from .base import BaseComparator
from .comparison_result import ComparisonResult


class DataComparator(BaseComparator):
    """Concrete class to handle comparison between two DataFrames."""

    def __init__(
        self,
        first_data: pd.DataFrame,
        second_data: pd.DataFrame,
        compare_cols: list[str] | None = None,
        feature_mapping: dict[str, str] | None = None,
    ) -> None:
        """Initialize the DataComparator with two DataFrames to compare.

        Args:
            first_data (pd.DataFrame): The first DataFrame to compare.
            second_data (pd.DataFrame): The second DataFrame to compare.
            compare_cols (list[str] | None, optional): A list of column names to compare between the DataFrames.
                                                      Defaults to None.
            feature_mapping (dict[str, str] | None, optional): A dictionary mapping column names in first_data
                                                            to their corresponding.
                                                            Defaults to None.

        """
        super().__init__(first_data, second_data, compare_cols, feature_mapping)

    def _check_column_existence(self, first_data_col: str, second_data_col: str) -> None:
        if first_data_col not in self.first_data.columns:
            if first_data_col in self.compare_cols:
                self.compare_cols.remove(first_data_col)
            msg = f"Error: Columns {first_data_col} not found in first_data DataFrames"
            raise ValueError(msg)
        if second_data_col not in self.second_data.columns:
            if second_data_col in self.compare_cols:
                self.compare_cols.remove(second_data_col)
            msg = f"Error: Columns {second_data_col} not found in second_data DataFrames"
            raise ValueError(msg)

    def _compare_unique_values(self, first_data_col: str, second_data_col: str) -> ComparisonResult:
        """Compare the unique values between two columns in the DataFrames.

        This method identifies missing values present in one DataFrame but not the other.

        Args:
            first_data_col (str): The name of the column in the first DataFrame.
            second_data_col (str): The name of the column in the second DataFrame.

        Returns:
            ComparisonResult: An object containing the comparison results, including
                the column names, number of unique values in each DataFrame, missing values in each DataFrame,
                and a flag indicating if the columns have the same set of unique values.

        """
        first_data_unique = set(self.first_data[first_data_col].unique())
        second_data_unique = set(self.second_data[second_data_col].unique())

        missing_in_first_data = list(second_data_unique.difference(first_data_unique))
        missing_in_second_data = list(first_data_unique.difference(second_data_unique))

        return ComparisonResult(
            first_data_col=first_data_col,
            second_data_col=second_data_col,
            comparison_type="Unique value comparison",
            is_matched= (len(missing_in_first_data) == 0) & (len(missing_in_second_data) == 0),
            first_data_count=len(first_data_unique),
            second_data_count=len(second_data_unique),
            missing={"missing_in_first_data": missing_in_first_data, "missing_in_second_data": missing_in_second_data},
            additional_info={"difference": len(first_data_unique) - len(second_data_unique)},
        )

    def _compare_length(self) -> ComparisonResult:
        """Compare the number of rows (length) between the two DataFrames.

        Returns:
            ComparisonResult: An object containing the comparison results, including
                the number of rows in each DataFrame and a
                flag indicating if the DataFrames have the same number of rows.

        """
        first_data_len = len(self.first_data)
        second_data_len = len(self.second_data)
        return ComparisonResult(
            comparison_type="Number of rows",
            first_data_count=first_data_len,
            second_data_count=second_data_len,
            is_matched= first_data_len == second_data_len,
            additional_info={"difference": len(self.first_data) - len(self.second_data)},
        )

    def _compare_unique_value(self) -> list[ComparisonResult]:
        """Compare unique values between columns.

        Returns:
            list[ComparisonResult]: A list of comparison results.

        """
        results = []
        for col in self.compare_cols:
            first_data_col = col
            second_data_col = self.feature_mapping[col]
            try:
                self._check_column_existence(first_data_col, second_data_col)
                results.append(self._compare_unique_values(first_data_col, second_data_col))
            except ValueError as e:
                error = ComparisonResult(
                    meta = [e],
                )
                results.append(error)
        return results

    def _compare_unique_grp(
        self,
        first_data_grp_col: str,
        first_data_col_val: str,
        second_data_grp_col: str,
        second_data_col_val: str,
    ) -> list[ComparisonResult]:
        """Compare unique values within grouped columns.

        Args:
            first_data_grp_col (str): The name of the grouping column in the first DataFrame.
            first_data_col_val (str): The name of the value column in the first DataFrame.
            second_data_grp_col (str): The name of the grouping column in the second DataFrame.
            second_data_col_val (str): The name of the value column in the second DataFrame.

        Returns:
            list[ComparisonResult]: A list of comparison results.

        """
        # Group by the specified columns and get unique values
        first_data_grouped = (
            self.first_data
            .groupby(first_data_grp_col)[first_data_col_val]
            .unique()
            .reset_index(name="first_data_values")
        )
        second_data_grouped = (
            self.second_data
            .groupby(second_data_grp_col)[second_data_col_val]
            .unique()
            .reset_index(name="second_data_values")
        )

        # Merge the grouped DataFrames on the grouping columns
        merged_data = (
            first_data_grouped
            .merge(
                second_data_grouped,
                left_on=first_data_grp_col,
                right_on=second_data_grp_col,
                how="inner")
        )
        if len(merged_data) == 0:
            return [ComparisonResult(
                first_data_col=first_data_grp_col,
                second_data_col=second_data_grp_col,
                comparison_type=f"grp -{first_data_col_val}",
                meta=[f"Error No common value in first_data and second_data of {first_data_grp_col}"],
            )]

        # Handle missing values
        merged_data["first_data_values"] = (
            merged_data["first_data_values"]
            .apply(lambda x: x if isinstance(x, list | np.ndarray) else [])
        )
        merged_data["second_data_values"] = (
            merged_data["second_data_values"]
            .apply(lambda x: x if isinstance(x, list | np.ndarray) else [])
        )

        # Compare the unique values
        def compare_values(row):  # noqa: ANN001, ANN202
            return sorted(row["first_data_values"]) == sorted(row["second_data_values"])

        merged_data["is_matched"] = merged_data.apply(compare_values, axis=1)
        merged_data["first_data_col"] = merged_data[first_data_grp_col].apply(lambda x: f"{first_data_grp_col} - {x!s}")
        merged_data["second_data_col"] = second_data_grp_col
        merged_data["comparison_type"] = f"Grp - {first_data_col_val}"
        merged_data["first_data_count"] = merged_data["first_data_values"].apply(lambda x: len(x))
        merged_data["second_data_count"] = merged_data["second_data_values"].apply(lambda x: len(x))
        merged_data["missing"] = merged_data.apply(lambda x: {
            "only_in_first_data": set(x["first_data_values"]).difference(x["second_data_values"]),
            "only_in_second_data": set(x["second_data_values"]).difference(x["first_data_values"]),
        }, axis=1)

        # Convert each row to a ComparisonResult instance
        results = []
        for _, row in merged_data.iterrows():
            result = ComparisonResult(
                first_data_col=row["first_data_col"],
                second_data_col=row["second_data_col"],
                comparison_type=row["comparison_type"],
                is_matched=row["is_matched"],
                first_data_count=row["first_data_count"],
                second_data_count=row["second_data_count"],
                missing=row["missing"],
            )
            results.append(result)
        return results

    def _has_unique_group_mismatch(self) -> list[ComparisonResult]:
        """Check for unique value mismatches across grouped columns.

        Returns:
            list[ComparisonResult]: A list of comparison results.

        """
        len_compare_cols = len(self.compare_cols)
        mismatches = []

        for i in range(len_compare_cols - 1):
            first_data_grp_col = self.compare_cols[i]
            second_data_grp_col = self.feature_mapping[first_data_grp_col]

            for j in range(i + 1, len_compare_cols):
                first_data_col_val = self.compare_cols[j]
                second_data_col_val = self.feature_mapping[first_data_col_val]

                result = self._compare_unique_grp(
                    first_data_grp_col,
                    first_data_col_val,
                    second_data_grp_col,
                    second_data_col_val,
                )
                mismatches.extend(result)
        return mismatches

    def compare(self, is_group_mismatch: bool| None=False) -> pd.DataFrame:
        """Perform a comprehensive comparison of the DataFrames.

        Args:
            is_group_mismatch (bool | None, optional): Whether to include group-based comparisons. Defaults to False.

        Returns:
            pd.DataFrame: A DataFrame containing the comparison results.

        """
        tests = [
            self._compare_length(),
            *self._compare_unique_value(),
            *(self._has_unique_group_mismatch() if is_group_mismatch else []),
        ]
        return pd.DataFrame(tests)

