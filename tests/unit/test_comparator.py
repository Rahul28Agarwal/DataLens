# ruff:noqa
import pandas as pd
import pytest

from ...DataLen.comparators.data_comparator import DataComparator

@pytest.fixture
def first_data():
     return pd.DataFrame({
        "A": [1, 2, 3, None, 5],
        "B": [0, 0, 0, 0, 0],
        "C": ["a", "b", "c", "d", "e"],
        "F": ["T", "G", "c", "d", "e"],
    })

@pytest.fixture
def second_data():
     return pd.DataFrame({
        "A": [1, 2, 3, None, 5],
        "B": [0, 0, 0, 0, 0],
        "C": ["a", "b", "c", "d", "e"],
        "D": ["T", "G", "c", "d", "e"],
    })
     
def test_basic_functionality(first_data, second_data):
    comparator = DataComparator(first_data, second_data)
    result = comparator.compare()
    assert type(result) == pd.DataFrame
    
def test_first_data_empty_dataframe(first_data, second_data):
    first_data = pd.DataFrame()
    with pytest.raises(ValueError, match="first_data cannot be empty"):
        comparator = DataComparator(first_data, second_data)
        result = comparator.compare()
        
def test_second_data_empty_dataframe(first_data, second_data):
    second_data = pd.DataFrame()
    with pytest.raises(ValueError, match="second_data cannot be empty"):
        comparator = DataComparator(first_data, second_data)
        result = comparator.compare()