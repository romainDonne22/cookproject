import pytest
import pandas as pd
from front import load_data, main

def test_load_data(monkeypatch):
    # Mock pd.read_csv to return a DataFrame
    def mock_read_csv(file):
        return pd.DataFrame({'note_moyenne': [4.5, 3.0, 5.0]})

    monkeypatch.setattr(pd, 'read_csv', mock_read_csv)
    data = load_data()
    assert not data.empty
    assert 'note_moyenne' in data.columns

    # Mock pd.read_csv to raise an exception
    def mock_read_csv_exception(file):
        raise Exception("File not found")

    monkeypatch.setattr(pd, 'read_csv', mock_read_csv_exception)
    data = load_data()
    assert data.empty

def test_main(monkeypatch):
    # Mock Streamlit functions
    def mock_title(title):
        assert title == "Recipe Manager"

    def mock_sidebar_selectbox(label, options):
        return "View Recipes"

    def mock_subheader(subheader):
        assert subheader == "View Recipes"

    def mock_data_frame(data):
        assert not data.empty

    monkeypatch.setattr(st, 'title', mock_title)
    monkeypatch.setattr(st.sidebar, 'selectbox', mock_sidebar_selectbox)
    monkeypatch.setattr(st, 'subheader', mock_subheader)
    monkeypatch.setattr(st, 'dataframe', mock_data_frame)

    # Mock load_data to return a DataFrame
    def mock_load_data():
        return pd.DataFrame({'note_moyenne': [4.5, 3.0, 5.0]})

    monkeypatch.setattr('front.load_data', mock_load_data)
    main()