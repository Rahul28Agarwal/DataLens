# ruff:noqa
import pandas as pd
import pytest

from ...datalens.investigator.numerical_investigator import NumericalDataInvestigator

@pytest.fixture
def sample_mixed_data():
    return pd.DataFrame({
        "A": [1, 2, 3, None, 5],
        "B": [0, 0, 0, 0, 0],
        "C": ["a", "b", "c", "d", "e"],
    })

@pytest.fixture
def non_numerical_dataframe() -> pd.DataFrame:
    """Provide DataFrame with no numerical columns.

    Returns:
        DataFrame containing only non-numerical data.

    """
    return pd.DataFrame({
        "C": ["a", "b", "c", "d", "e"],
        "D": [True, False, True, True, False],
    })

def test_descibe_columns_with_numerical_data(sample_mixed_data):
    investigator = NumericalDataInvestigator(sample_mixed_data)
    result = investigator.describe_columns()

    # Check if the result DataFrame has the expected columns
    expected_columns = ["Column", "Data Type", "Count", "Missing", "Zeros",
                        "Unique", "Min", "25%", "Mean", "Median",
                        "75%", "90%", "99%", "Max", "StdDev",
                        "Skew", "Kurtosis", "IQR"]

    assert all(col in result.columns for col in expected_columns)

def test_describe_columns_empty_dataframe(non_numerical_dataframe):
    investigator = NumericalDataInvestigator(non_numerical_dataframe)

    with pytest.raises(ValueError, match="DataFrame has no numerical data."):
        investigator.describe_columns()
