"""Tests for DataCleaner."""
 
import pandas as pd
 
from src.config import REQUIRED_COLUMNS
from src.data_cleaner import DataCleaner
 
 
def test_data_cleaner_fills_missing_values_and_drops_duplicates():
    row = {
        "longitude": -122.0,
        "latitude": 37.0,
        "housing_median_age": 20,
        "total_rooms": 1000,
        "total_bedrooms": None,
        "population": 500,
        "households": 200,
        "median_income": 3.5,
        "median_house_value": 200000,
        "ocean_proximity": "NEAR BAY",
    }
 
    dataframe = pd.DataFrame([row, row.copy()])
 
    cleaner = DataCleaner(dataframe)
    cleaned = cleaner.clean(REQUIRED_COLUMNS)
 
    assert cleaned.isna().sum().sum() == 0
    assert len(cleaned) == 1
