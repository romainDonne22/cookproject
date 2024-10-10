import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns 
import matplotlib

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
    
    # Calculer la moyenne totale des notes
    nub_ratings = len(data['rating'])
    nub_users = len(data['user_id'])
    mean_ratings = data['rating'].mean()
    print("Le nombre total d'utilisateurs est : ", nub_users)
    print("Le nombre total de notes est : ", nub_ratings)
    print("La moyenne totale des notes est : ", mean_ratings)

    # Calculer la répartition des notes
    ratings_count = data['rating'].value_counts().sort_index()

    # Créer un DataFrame pour les notes
    ratings_df = pd.DataFrame({'rating': ratings_count.index, 'count': ratings_count.values})

    # Tracer un graphique en barres de la répartition des notes avec Seaborn
    plt.figure(figsize=(10, 6))
    sns.barplot(x='rating', y='count', data=ratings_df, palette='viridis', edgecolor='black')
    plt.title('Répartition des notes')
    plt.xlabel('Notes')
    plt.ylabel('Nombre de personnes')
    # Enregistrer le graphique dans un fichier
    plt.savefig('repartition_des_notes.png')

# Analyse the dataset RAW_recipes
recipe = pd.read_csv("../data/RAW_recipes.csv")
#datasetAnalysis(recipe)

# Analyse the dataset PP_users
datasetAnalysis(pd.read_csv("../data/RAW_interactions.csv"))
analyseStars(pd.read_csv("../data/RAW_interactions.csv"))