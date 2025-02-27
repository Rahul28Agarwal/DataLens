import pytest
import pandas as pd
from ...DataLen.components.bivariate_data_investigator import BivariateDataInvestigator

def test_empty_dataframe():
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="The provided DataFrame is empty"):
        bivariate_investigator = BivariateDataInvestigator(df)
    
@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'A': [1, 2, 3, None, 5],
        'B': [0, 0, 0, 0, 0],
        'C': ['a', 'b', 'c', 'd', 'e'],
        'F': ['T', 'G', 'c', 'd', 'e']
    })

def test_dataframe_not_column_exist(sample_data):
    bivariate_investigator = BivariateDataInvestigator(sample_data)
    column = 'D'
    
    with pytest.raises(ValueError, match=f"Column {column} does not exist"):
        bivariate_investigator.analyze_bivariate_relationship('A', column)
        
def test_analyze_numerical_categorical_relationship(sample_data):
    bivariate_investigator = BivariateDataInvestigator(sample_data)
    assert bivariate_investigator.analyze_bivariate_relationship("A", "C") is None

def test_analyze_categorical_relationship(sample_data):
    bivariate_investigator = BivariateDataInvestigator(sample_data)
    assert bivariate_investigator.analyze_bivariate_relationship("F", "C") is None