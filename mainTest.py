import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def datasetAnalysis(data):
    # Print the first three rows of the data
    print(data.head(3))

    # Print the shape of the data
    num_ligne = data.shape[0]
    print("Le nombre de lignes dans le dataset est: ", num_ligne)
    num_columns = data.shape[1]
    print("Le nombre de colonnes dans le dataset est : ", num_columns)

    # Print the missing values
    print("Les missing values sont : ", data.isnull().sum().sum())
    print("Par variable", data.isna().sum())

    # Overview of the columns with missing values
    missing_data = data[data.isnull().any(axis=1)]
    if not missing_data.empty:
        print("10 observations contenant des valeurs manquantes :")
        print(missing_data.head(10))
    else:
        print("Aucune valeur manquante trouvée dans le dataset.")

    # Specific cleaning for RAW_recipes dataset
    if dataset_name == "RAW_recipes.csv":
        print("Nettoyage spécifique pour le dataset RAW_recipes")
        # Remplacer les valeurs manquantes par "missing"
        data_cleaned = data.fillna("missing")
        print("Dataset nettoyé :")
        print(data_cleaned.head(3))
        # Vous pouvez ajouter d'autres opérations de nettoyage spécifiques ici


# Import the data and print the first three rows
# recipe = pd.read_csv("../data/RAW_recipes.csv")
recipe = pd.read_csv("../data/PP_users.csv")
# Analyse the dataset RAW_recipes
datasetAnalysis(recipe)
