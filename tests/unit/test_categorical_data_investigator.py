import pandas as pd
import pytest
from ...datalens.investigator.categorical_investigator import CategoricalDataInvestigator

@pytest.fixture
def sample_mixed_data():
    return pd.DataFrame({
        'A': [1, 2, 3, None, 5],
        'B': [0, 0, 0, 0, 0],
        'C': ['a', 'b', 'c', 'd', 'e']
    })

@pytest.fixture
def non_categorical_dataframe() -> pd.DataFrame:
    return pd.DataFrame({
        'A': [1, 2, 3, None, 5],
        'B': [0, 0, 0, 0, 0],
    })

def test_descibe_columns_with_categorical_data(sample_mixed_data):
    investigator = CategoricalDataInvestigator(sample_mixed_data)
    result = investigator.describe_columns()

    # Check if the result DataFrame has the expected columns
    expected_columns = ['Column', 'Data Type', 'Count', 'Missing', 'Unique', 
                        'Mode', 'Mode Frequency', 'Mode Percentage', 'Second Mode', 'Entropy', 
                        ]
    
    assert all(col in result.columns for col in expected_columns)

def test_describe_columns_without_categorical_column(non_categorical_dataframe):
    investigator = CategoricalDataInvestigator(non_categorical_dataframe)

    with pytest.raises(ValueError, match="DataFrame has no categorical data."):
        investigator.describe_columns()