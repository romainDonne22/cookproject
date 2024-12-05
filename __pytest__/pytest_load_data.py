import sys
import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from matplotlib.figure import Figure
from statsmodels.regression.linear_model import RegressionResultsWrapper
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from collections import Counter
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C:/Users/camil/OneDrive/Bureau/MS_BGD/Git_cookproject/cookproject')))

# Importer les fonctions du script principal
from rating_recipe_correlation_analysis import (
    load_csv,
    append_csv,
    merged_data,
    check_duplicates,
    drop_columns,
    dropNa,
    fillNa,
    remove_outliers,
    calculate_outliers,
    calculate_quartile,
    separate_bad_good_ratings,
    plot_bad_ratings_distributions,
    saisonnalite,
    boxplot_df,
    rating_distribution,
    OLS_regression,
    preprocess_text,
    get_most_common_words,
    extractWordFromTUpple,
    uniqueTags,
    time_per_step,
    rating_isContributor,
    plot_distributionIsContributor,
    create_data_part1,
    create_data_part2,
    create_dfuser_profiles,
    dflog,
    plot_distribution,
    plot_correlation_matrix
)

# Simuler un dataframe pour les tests
@pytest.fixture
def sample_dataframe():
    data = {
        'id': [1, 2, 3, 4, 5],
        'value': [10, 15, 10, 20, 25],
        'text': ['Hello World', 'Test string', None, 'Another test', 'Final test'],
        'category': [1, 1, 2, 2, 3]
    }
    return pd.DataFrame(data)

@pytest.fixture
def second_dataframe():
    data = {
        'id': [4, 5, 6],
        'extra': ['A', 'B', 'C']
    }
    return pd.DataFrame(data)

# Test de chargement CSV
def test_load_csv(tmp_path):
    """
    Test loading a CSV file into a DataFrame.
    """
    file_path = tmp_path / "test.csv"
    df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['A', 'B', 'C']})
    df.to_csv(file_path, index=False)
    
    loaded_df = load_csv(file_path)
    pd.testing.assert_frame_equal(df, loaded_df)

# Test de concaténation CSV
def test_append_csv(tmp_path):
    """
    Test appending multiple CSV files into a single DataFrame.
    """
    file1 = tmp_path / "file1.csv"
    file2 = tmp_path / "file2.csv"
    df1 = pd.DataFrame({'col1': [1, 2], 'col2': ['A', 'B']})
    df2 = pd.DataFrame({'col1': [3, 4], 'col2': ['C', 'D']})
    df1.to_csv(file1, index=False)
    df2.to_csv(file2, index=False)
    
    combined_df = append_csv(file1, file2)
    expected_df = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': ['A', 'B', 'C', 'D']})
    pd.testing.assert_frame_equal(combined_df, expected_df)

# Test de jointure de données
def test_merged_data(sample_dataframe, second_dataframe):
    """
    Test merging two DataFrames.
    """
    merged_df = merged_data(sample_dataframe, second_dataframe, 'id', 'id', 'inner')
    assert len(merged_df) == 2  # 2 lignes correspondantes

# Test de détection des doublons
def test_check_duplicates(sample_dataframe):
    """
    Test checking for duplicate rows in a DataFrame.
    """
    sample_dataframe.loc[5] = sample_dataframe.iloc[0]  # Ajouter un doublon
    num_duplicates = check_duplicates(sample_dataframe)
    assert num_duplicates == 1

# Test suppression de colonnes
def test_drop_columns(sample_dataframe):
    """
    Test dropping specified columns from a DataFrame.
    """
    drop_columns(sample_dataframe, ['text'])
    assert 'text' not in sample_dataframe.columns

# Test suppression des valeurs manquantes
def test_dropNa(sample_dataframe):
    """
    Test dropping rows with missing values in specified columns.
    """
    initial_length = len(sample_dataframe)
    dropNa(sample_dataframe, ['text'])
    assert len(sample_dataframe) < initial_length

# Test remplissage des valeurs manquantes
def test_fillNa(sample_dataframe):
    """
    Test filling missing values in a specified column with a given value.
    """
    fillNa(sample_dataframe, 'text', 'default')
    assert 'default' in sample_dataframe['text'].values

# Test suppression des outliers
def test_remove_outliers():
    """
    Test removing outliers from specified columns in a DataFrame.
    """
    data = pd.DataFrame({
        'col': [10, 20, 30, 40, 1000]
    })
    cleaned_data = remove_outliers(data, ['col'])
    
    # Calculer les bornes pour vérifier le test
    Q1 = data['col'].quantile(0.15)
    Q3 = data['col'].quantile(0.85)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    
    # Vérifier que toutes les valeurs sont inférieures ou égales à la borne supérieure
    assert cleaned_data['col'].max() <= upper_bound

# Test de nettoyage de texte
def test_preprocess_text():
    """
    Test preprocessing text by converting to lowercase, removing punctuation, special characters, and stop words.
    """
    text = "Hello, this is a Test 123!"
    cleaned_text = preprocess_text(text)
    assert cleaned_text == "hello test"

# Test calcul des outliers
def test_calculate_outliers(sample_dataframe):
    """
    Test calculating outliers for specified numerical columns in a DataFrame.
    """
    outlier_summary = calculate_outliers(sample_dataframe, ['value'])
    assert 'value' in outlier_summary.index
    assert outlier_summary.loc['value', 'Outlier Count'] >= 0

# Test pour la fonction dflog
def test_dflog(sample_dataframe):
    """
    Test applying log transformation to a specified column in a DataFrame.
    """
    new_column = dflog(sample_dataframe, 'value')
    assert 'newcolumn' in sample_dataframe.columns
    assert np.allclose(new_column, np.log1p(sample_dataframe['value']))

# Test pour la fonction plot_distribution
def test_plot_distribution(sample_dataframe):
    """
    Test plotting the distribution of a specified column in a DataFrame.
    """
    fig = plot_distribution(sample_dataframe, 'value', 'Distribution de la valeur')
    assert isinstance(fig, plt.Figure)

# Test pour la fonction plot_correlation_matrix
def test_plot_correlation_matrix(sample_dataframe):
    """
    Test plotting the correlation matrix of specified columns in a DataFrame.
    """
    sample_dataframe['value2'] = sample_dataframe['value'] * 2  # Ajouter une deuxième colonne pour la corrélation
    fig = plot_correlation_matrix(sample_dataframe, ['value', 'value2'], 'Matrice de corrélation')
    assert isinstance(fig, plt.Figure)

@pytest.fixture
def sample_data():
    data = {
        "minutes": [5, 10, 15, 20, 25],
        "n_steps": [3, 5, 7, 9, 11],
        "n_ingredients": [4, 5, 6, 7, 7],
        "ratings": [1.5, 2.5, 3.5, 4.5, 5.0],
        "day_of_week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        "month": ["January", "February", "March", "April", "May"]
    }
    return pd.DataFrame(data)


# Tests
def test_calculate_quartile(sample_data):
    """
    Test calculate_quartile function.
    """
    result = calculate_quartile(sample_data, "minutes", 0.5)
    assert result == 15, "Le calcul du quantile est incorrect."

def test_separate_bad_good_ratings(sample_data):
    """
    Test separate_bad_good_ratings function.
    """
    bad, good = separate_bad_good_ratings(sample_data, 3.0, "ratings")
    assert len(bad) == 2, "Le nombre de mauvaises notes est incorrect."
    assert len(good) == 3, "Le nombre de bonnes notes est incorrect."

def test_plot_bad_ratings_distributions(sample_data):
    """
    Test plot_bad_ratings_distributions function.
    """
    bad, good = separate_bad_good_ratings(sample_data, 3.0, "ratings")
    fig = plot_bad_ratings_distributions(bad, good)
    assert isinstance(fig, Figure), "Le retour n'est pas une figure matplotlib."

def test_saisonnalite(sample_data):
    """
    Test saisonnalite function.
    """
    fig = saisonnalite(sample_data)
    assert isinstance(fig, Figure), "Le retour n'est pas une figure matplotlib."

def test_boxplot_df(sample_data):
    """
    Test boxplot_df function.
    """
    fig = boxplot_df([sample_data["minutes"], sample_data["n_steps"]])
    assert isinstance(fig, Figure), "Le retour n'est pas une figure matplotlib."

def test_rating_distribution(sample_data):
    """
    Test rating_distribution function.
    """
    fig, comparison_df = rating_distribution(
        sample_data, "minutes", "ratings", 10, (10, 20), 20
    )
    assert isinstance(fig, Figure), "Le retour n'est pas une figure matplotlib."
    assert isinstance(comparison_df, pd.DataFrame), "Le retour n'est pas un DataFrame."

def test_OLS_regression(sample_data):
    """
    Test OLS_regression function.
    """
    X = sample_data[["minutes", "n_steps"]]
    y = sample_data["ratings"]
    model = OLS_regression(X, y)
    assert isinstance(model, RegressionResultsWrapper), "Le modèle n'est pas une régression OLS valide."


# Fixtures
@pytest.fixture
def sample_text():
    return "This is an Example TEXT with numbers 123 and Punctuation!!!"

@pytest.fixture
def sample():
    data = {
        "tags": ["tag1 tag2 tag3", "tag2 tag3", "tag3", None],
        "minutes": [10, 20, 30, 40],
        "n_steps": [2, 4, 6, 8],
        "rating": [1, 2, 3, 4],
        "contributor_id": [1, 2, 1, 3],
        "user_id": [1, 2, 3, 4]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_word_tuples():
    return [("apple", 5), ("banana", 3), ("cherry", 7)]

def test_preprocess_text(sample_text):
    """
    Test preprocess_text function.
    """
    result = preprocess_text(sample_text)
    expected = "example text numbers punctuation"
    assert result == expected, "Le texte prétraité est incorrect."

def test_get_most_common_words(sample):
    """
    Test get_most_common_words function.
    """
    result = get_most_common_words(sample["tags"])
    assert isinstance(result, list), "Le résultat doit être une liste."
    assert len(result) > 0, "Le résultat ne doit pas être vide."
    assert result[0][0] == "tag3", "Le mot le plus fréquent est incorrect."

def test_extractWordFromTUpple(sample_word_tuples):
    """
    Test extractWordFromTUpple function.
    """
    result = extractWordFromTUpple(sample_word_tuples)
    assert isinstance(result, set), "Le résultat doit être un ensemble (set)."
    assert "apple" in result, "Le mot 'apple' devrait être présent."
    assert "banana" in result, "Le mot 'banana' devrait être présent."

def test_uniqueTags():
    """
    Test uniqueTags function.
    """
    list1 = {"tag1", "tag2", "tag3"}
    list2 = {"tag2", "tag3"}
    result = uniqueTags(list1, list2)
    assert result == {"tag1"}, "Les tags uniques sont incorrects."

def test_time_per_step(sample):
    """
    Test time_per_step function.
    """
    fig = time_per_step(sample, "minutes", "n_steps")
    assert isinstance(fig, Figure), "Le résultat doit être une figure matplotlib."

def test_rating_isContributor(sample):
    """
    Test rating_isContributor function.
    """
    fig = rating_isContributor(sample, "contributor_id")
    assert isinstance(fig, Figure), "Le résultat doit être une figure matplotlib."
    
@pytest.fixture
def sample_plot_distributionIsContributor():
    """
    Fixture for sample data for plot_distributionIsContributor.

    Returns:
        DataFrame: Sample data for testing.
    """
    data = {
        'is_contributor': [0, 1, 0, 1, 0],
        'ratings': [1.5, 2.5, 3.5, 4.5, 5.0]
    }
    return pd.DataFrame(data)

def test_plot_distributionIsContributor(sample_plot_distributionIsContributor):
    """
    Test plot_distributionIsContributor function.
    """
    fig = plot_distributionIsContributor(sample_plot_distributionIsContributor, 'is_contributor', 'ratings')
    assert isinstance(fig, plt.Figure), "Le retour n'est pas une figure matplotlib."

# # Tests pour create_data_part1
# def test_create_data_part1(mocker):
#     mocker.patch("your_module.load_csv", return_value=pd.DataFrame())
#     mocker.patch("your_module.append_csv", return_value=pd.DataFrame())
#     mocker.patch("your_module.merged_data", return_value=pd.DataFrame())
#     mocker.patch("your_module.remove_outliers", return_value=pd.DataFrame())
#     result = create_data_part1()
#     assert isinstance(result, pd.DataFrame), "Le résultat doit être un DataFrame."

# # Tests pour create_data_part2
# def test_create_data_part2(mocker):
#     mocker.patch("your_module.append_csv", return_value=pd.DataFrame())
#     mocker.patch("your_module.merged_data", return_value=pd.DataFrame())
#     mocker.patch("your_module.remove_outliers", return_value=pd.DataFrame())
#     result = create_data_part2()
#     assert isinstance(result, pd.DataFrame), "Le résultat doit être un DataFrame."


def test_create_dfuser_profiles(sample):
    """
    Test create_dfuser_profiles function.
    """
    result = create_dfuser_profiles(sample)
    assert isinstance(result, pd.DataFrame), "Le résultat doit être un DataFrame."
    assert "num_recipes_rated" in result.columns, "La colonne 'num_recipes_rated' devrait être présente."
    assert "mean_rating" in result.columns, "La colonne 'mean_rating' devrait être présente."

# Simuler un dataframe pour les tests
@pytest.fixture
def sample_rating_distribution():
    data = {
        'variable': [1, 2, 3, 4, 5],
        'ratings': [1.5, 2.5, 3.5, 4.5, 5.0]
    }
    return pd.DataFrame(data)


def test_rating_distribution(sample_rating_distribution):
    """
    Test rating_distribution function.
    """
    fig, comparison_df = rating_distribution(
        sample_rating_distribution,
        variable='variable',
        rating_var='ratings',
        low_threshold=2,
        mean_range=(2, 4),
        high_threshold=4
    )
    assert isinstance(fig, plt.Figure), "Le retour n'est pas une figure matplotlib."
    assert isinstance(comparison_df, pd.DataFrame), "Le retour n'est pas un DataFrame."
    expected_columns = ['Catégorie élevée', 'Catégorie moyenne', 'Catégorie basse']
    assert list(comparison_df.columns) == expected_columns
    expected_index = ['Less than 2', '2 to 3', '3 to 4', '4 to 5']
    assert list(comparison_df.index) == expected_index