import pandas as pd
import logging

class DataValidator:
    """
    Class for validating Pandas DataFrames.
    """
    def __init__(self, data: pd.DataFrame):
        """
        Initializes the DataValidator with the DataFrame to be validated.

        Args:
            data (pd.DataFrame): The DataFrame to validate.
        """
        self.data = data


    def validate_is_dataframe(self):
        """
        Validates that the data is a Pandas DataFrame.

        Raises:
            TypeError: If the data is not a DataFrame.
        """
        if not isinstance(self.data, pd.DataFrame):
            msg = "Provided data is not a pandas DataFrame"
            logging.error(msg)
            raise TypeError(msg)

    def validate_not_empty(self):
        """
        Validates that the DataFrame is not empty.

        Raises:
            ValueError: If the DataFrame is empty.
        """
        if self.data.empty:
            msg = "The provided DataFrame is empty"
            logging.warning(msg)
            raise ValueError(msg)

    def validate_duplicate_columns(self):
        """
        Validates that the DataFrame does not contain duplicate columns.

        Raises:
            ValueError: If the DataFrame contains duplicate columns.
        """
        if len(self.data.columns) != len(set(self.data.columns)):
            msg = "DataFrame contains duplicate columns"
            logging.error(msg)
            raise ValueError(msg)

    def validate_column(self, column: str) -> None:
        """
        Validates that a given column exists in the DataFrame.

        Args:
            column (str): The name of the column to validate.

        Raises:
            ValueError: If the column does not exist in the DataFrame.
        """
        if column not in self.data.columns:
            msg = f"Column {column} does not exist"
            logging.error(msg)
            raise ValueError(msg)
