import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns 
import ast

def datasetAnalysis(data):
    print("-------------------")
    print("Dataset check ")
    print("-------------------")
    # Print the first three rows of the data
    print(data.head(3))

    # Print the shape of the data
    num_ligne = data.shape[0]
    print("Le nombre de lignes dans le dataset est: ", num_ligne)
    num_columns = data.shape[1]
    print("Le nombre de colonnes dans le dataset est : ", num_columns)

    # Print the data types of the columns
    listeColonne = data.columns.tolist()
    listeColonneType = data.dtypes.tolist()
    print("Le type des colonnes est : ", listeColonne, listeColonneType)

    # Print the missing values
    print("Les missing values sont : ", data.isnull().sum().sum())
    print("Par varaible",data.isna().sum())



def analyseStars(data):
    print("-------------------")
    print("Stars Analyse ")
    print("-------------------")
    
    # Convertir les valeurs de la colonne 'ratings' en listes de nombres
    data['ratings'] = data['ratings'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Aplatir toutes les listes en une seule liste
    all_ratings = [item for sublist in data['ratings'] for item in sublist]
    
    # Calculer la moyenne totale des notes
    nub_ratings = len(all_ratings)
    nub_users = len(data['u'])
    mean_ratings = sum(all_ratings) / nub_ratings
    print("Le nombre total d'utilisateurs est : ", nub_users)
    print("Le nombre total de notes est : ", nub_ratings)
    print("La moyenne totale des notes est : ", mean_ratings)

    plt.figure()
    plt.scatter(all_ratings, data['u'])
    plt.title('Investment data')
    plt.xlabel('Gross National Product')
    plt.ylabel('Investment')
    plt.show()

# Analyse the dataset RAW_recipes
recipe = pd.read_csv("../data/RAW_recipes.csv")
#datasetAnalysis(recipe)

# Analyse the dataset PP_users
datasetAnalysis(pd.read_csv("../data/PP_users.csv"))
analyseStars(pd.read_csv("../data/PP_users.csv"))