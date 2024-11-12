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
    print("Par variable", data.isna().sum())

    # Overview of the columns with missing values
    missing_data = data[data.isnull().any(axis=1)]
    if not missing_data.empty:
        print("10 observations contenant des valeurs manquantes :")
        print(missing_data.head(10))
    else:
        print("Aucune valeur manquante trouvée dans le dataset.")

    # Checking distribution of the data for each variables
    print(data.describe())

    for col in data.select_dtypes(include=[np.number]).columns:
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
        print(f"Nombre d'outliers pour la variable "f"{col} : {outliers.shape[0]}")

    # Convertir les colonnes contenant "id" en type 'category'
    try:
        categorical_columns = data.select_dtypes(include=['category']).columns
        if categorical_columns.empty:
            raise ValueError("pas de colonnes categorical dans le dataset")
        for col in categorical_columns:
            if "id" in col.lower() and data[col].dtype == 'int64':
                data[col] = data[col].astype('category')
                print(f"Colonne {col} convertie en type 'category'")
    except ValueError as e:
        print(e)


def analyseStars(data):
    print("-------------------")
    print("Stars Analyse ")
    print("-------------------")

    # Calculer la moyenne totale des notes
    nub_ratings = len(data['rating'])
    nub_users = data['user_id'].nunique()
    mean_ratings = data['rating'].mean()
    print("Le nombre total d'utilisateurs est : ", nub_users)
    print("Le nombre total de notes est : ", nub_ratings)
    print("La moyenne totale des notes est : ", mean_ratings)
    print("Le nombre moyen de notes émis par utilisateur est : ", nub_ratings/nub_users)

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
    #plt.savefig('repartition_des_notes.png')

    #  lister les mauvaises notes cad <=2
    tabBadRating = []
    tabBadRating = data[data['rating'] <= 2]
    print("Le nombre de mauvaises notes <= 2 est : ", len(tabBadRating))
    print("Soit environ : ", len(tabBadRating)/nub_ratings*100, "%")
    print(tabBadRating.head(3))
    print("Les missing values sont : ", tabBadRating.isnull().sum().sum())
    print("Par varaible", tabBadRating.isna().sum())

    # Calculer les statistiques pour chaque recette
    grouped = data.groupby('recipe_id').agg(
        nb_user=('user_id', 'nunique'),
        note_moyenne=('rating', 'mean'),
        note_mediane=('rating', 'median'),
        note_q1=('rating', lambda x: x.quantile(0.25)),
        note_q2=('rating', lambda x: x.quantile(0.50)),
        note_q3=('rating', lambda x: x.quantile(0.75)),
        note_q4=('rating', lambda x: x.quantile(1.00)),
        note_max=('rating', 'max'),
        note_min=('rating', 'min'),
        nb_note_lt_5=('rating', lambda x: (x < 5).sum()),
        nb_note_eq_5=('rating', lambda x: (x == 5).sum())
    ).reset_index()

    # Afficher le nouveau DataFrame
    print(grouped.head())
    # Enregistrer le nouveau DataFrame dans un fichier CSV
    grouped.to_csv('recette_statistiques.csv', index=False)
    # Trouver la ligne qui a le plus de nb_user
    max_nb_user_row = grouped.loc[grouped['nb_user'].idxmax()]
    print("La ligne avec le plus grand nombre d'utilisateurs :")
    print(max_nb_user_row)

path_data = "../../data/RAW_interactions.csv"
datasetAnalysis(pd.read_csv(path_data))
analyseStars(pd.read_csv(path_data))

# mediane
# quantile
# nb de note pas userid, notamment voir si un user a mis plusierus mauvaises notes, à mis plusieurs avis sur une recette
# review mettre des tags
