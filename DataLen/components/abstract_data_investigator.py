from abc import ABC, abstractmethod
import pandas as pd
from ..core.data_validator import DataValidator

class AbstractDataInvestigator(ABC):
    def __init__(self, data: pd.DataFrame) -> None:
        self._data = None
        self.data = data
        self.validator = DataValidator(data)

    @property
    def data(self) -> pd.DataFrame:
        return self._data
    
    @data.setter
    def data(self, data: pd.DataFrame) -> None:
        """Setter  for the data property with validation

        Args:
            data (pd.DataFrame): Input pandas DataFrame
        """
        self.validator = DataValidator(data)
        self.validator.validate_is_dataframe()
        self.validator.validate_not_empty()

        self._data = data
        self.data_length = self._data.shape[0]
        self.numeric_columns = self._data.select_dtypes(include=["number"]).columns
        self.categorical_columns = self._data.select_dtypes(exclude=["number"]).columns