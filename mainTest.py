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

    # Checking distribution of the data for each variables
    data.describe()

    for col in data.columns:
        # Checking outliers
        # Calcul des quartiles
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1  # Intervalle interquartile
        # Définir les limites pour détecter les outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Filtrer les outliers
        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
        print(f"Nombre d'outliers pour la variable {
              col} : {outliers.shape[0]}")

    # Convertir les colonnes contenant "id" en type 'category'
        if "id" in col.lower() and data[col].dtype == 'int64':
            data[col] = data[col].astype('category')
            print(f"Colonne {col} convertie en type 'category'")

    # Specific cleaning for RAW_recipes dataset
    if data == "../data/RAW_recipes.csv":
        print("Nettoyage spécifique pour le dataset RAW_recipes")
        # Remplacer les valeurs manquantes par "missing"
        data_cleaned = data.fillna("missing")
        print("Dataset nettoyé :")
        print(data_cleaned.head(3))

        # Remplacer les valeurs min de la colonne minutes par des valeurs aléatoires entre 1 et 15 si la valeur est 0 et que le tag "15 min or less" est présent
        def replace_minutes(row):
            if row['minutes'] == 0 and "15-minutes-or-less" in row['tags']:
                return np.random.randint(1, 16)
            return row['minutes']

        data_cleaned['minutes'] = data_cleaned.apply(replace_minutes, axis=1)

        print("Dataset nettoyé :")
        print(data_cleaned.head(3))

        # Removing max outliers from col minutes
        time_month = 30*24*60
        data_cleaned.loc[data_cleaned['minutes'] > time_month].head(5)
        idx = data_cleaned.index[data_cleaned['minutes'] > time_month].tolist()
        # Then remove the row using the list of indices
        data_woa = data_cleaned.drop(idx)
        print("In total, we removed {} observations considered as outliers".format(
            data_cleaned.shape[0]-data_woa.shape[0]))

        # Nutrition score processing
        data_woa[['calories', 'total fat (%)', 'sugar (%)', 'sodium (%)', 'protein (%)', 'saturated fat (%)',
                  'carbohydrates (%)']] = data_woa.nutrition.str.split(",", expand=True)
        data_woa['calories'] = data_woa['calories'].apply(
            lambda x: x.replace('[', ''))
        data_woa['carbohydrates (%)'] = data_woa['carbohydrates (%)'].apply(
            lambda x: x.replace(']', ''))

        # Conversion en float
        data_woa[['calories', 'total fat (%)', 'sugar (%)', 'sodium (%)', 'protein (%)', 'saturated fat (%)', 'carbohydrates (%)']] = data_woa[[
            'calories', 'total fat (%)', 'sugar (%)', 'sodium (%)', 'protein (%)', 'saturated fat (%)', 'carbohydrates (%)']].astype(float)
        data_woa.describe()

        print("Dataframe avec la colonne Nutrition processed :")
        print(data_woa.head(3))

        # Supprimer les lignes avec des valeurs de calories anormales
        data_woa = data_woa[data_woa['calories'] <= 10000]
        # Réinitialiser l'index si nécessaire
        # data_woa.reset_index(drop=True, inplace=True)


# Import the data and print the first three rows
# recipe = pd.read_csv("../data/RAW_recipes.csv")
recipe = pd.read_csv("../data/PP_users.csv")
# Analyse the dataset RAW_recipes
datasetAnalysis(recipe)
