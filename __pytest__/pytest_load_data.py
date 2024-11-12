import sys
import os
import pytest
import pandas as pd


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C:/Users/camil/OneDrive/Bureau/MS_BGD/Git_cookproject/cookproject')))

from front import load_data


def test_load_data_valid_file(tmp_path):
    # Create a temporary CSV file
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("col1,col2\n1,2\n3,4")

    # Load the data
    data = load_data(csv_file)

    # Check if the data is loaded correctly
    assert not data.empty
    assert list(data.columns) == ["col1", "col2"]
    assert data.shape == (2, 2)

def test_load_data_non_existent_file():
    # Load data from a non-existent file
    data = load_data("non_existent_file.csv")

    # Check if the data is empty
    assert data.empty

def test_load_data_invalid_file_format(tmp_path):
    # Create a temporary invalid file
    invalid_file = tmp_path / "test.txt"
    invalid_file.write_text("This is not a CSV file")

    # Load the data
    data = load_data(invalid_file)

    # Check if the data is empty
    assert data.empty