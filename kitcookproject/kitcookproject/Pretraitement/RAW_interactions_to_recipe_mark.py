import pandas as pd
import numpy as np

def datasetPretraitement(path_data):
    # Load the data
    data = pd.read_csv(path_data)
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

    # Calculer la moyenne totale des notes
    nub_ratings = len(data['rating'])
    nub_users = data['user_id'].nunique()
    mean_ratings = data['rating'].mean()
    print("Le nombre total d'utilisateurs est : ", nub_users)
    print("Le nombre total de notes est : ", nub_ratings)
    print("La moyenne totale des notes est : ", mean_ratings)
    print("Le nombre moyen de notes émis par utilisateur est : ", nub_ratings/nub_users)

    #  lister les mauvaises notes cad <=2
    tabBadRating = []
    tabBadRating = data[data['rating'] <= 2]
    print("Le nombre de mauvaises notes <= 2 est : ", len(tabBadRating))
    print("Soit environ : ", len(tabBadRating)/nub_ratings*100, "%")
    print(tabBadRating.head(3))
    print("Les missing values sont : ", tabBadRating.isnull().sum().sum())
    print("Par variable", tabBadRating.isna().sum())

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
    grouped.head()
    # Trouver la ligne qui a le plus de nb_user
    max_nb_user_row = grouped.loc[grouped['nb_user'].idxmax()]
    print("La ligne avec le plus grand nombre d'utilisateurs :")
    print(max_nb_user_row)
    # Enregistrer le nouveau DataFrame dans un fichier parquet
    grouped.to_parquet('recipe_mark.parquet', index=False)
    print("Le fichier recipe_mark.parquet a été enregistré avec succès.")


path_data = "../data/RAW_interactions.csv"
datasetPretraitement(path_data)
