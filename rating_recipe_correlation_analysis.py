import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from collections import Counter

def load_csv(fichier):
    try:
        data = pd.read_csv(fichier)
        return data
    except Exception as e:
        print(f"Failed to load data: {e}")
        return pd.DataFrame()
    
def append_csv(*files):
    df_list = []
    for file in files:
        df = pd.read_csv(file)
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)
    
def merged_data(data1, data2, LO, RO, H):
    try:
        merged_df = pd.merge(data1, data2, left_on=LO, right_on=RO, how=H)
        return merged_df
    except Exception as e:
        print(f"Failed to load data: {e}")
        return pd.DataFrame()

def check_duplicates(df):
    num_duplicates = df.duplicated().sum()
    return num_duplicates

def drop_columns(df, columns_to_drop):
    df.drop(columns_to_drop, axis=1, inplace=True)

def dropNa(df, columns_to_drop):
    df.dropna(subset=columns_to_drop)

def fillNa(df, column_to_fill, value):
    df[column_to_fill].fillna(value)

def rename_columns(df, new_column_names):
    df.columns = new_column_names

def dflog(df, column):
    df['newcolumn'] = np.log1p(df[column])
    return df['newcolumn']

def plot_distribution(df, column, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter([1, 2, 3], [1, 2, 3])
    plt.hist(df[column], bins=20)
    plt.title(title)
    plt.xlabel(title)
    plt.ylabel('Fréquence')
    plt.tight_layout()
    return fig

def plot_correlation_matrix(df, columns, title):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter([1, 2, 3], [1, 2, 3])
    correlation = df[columns].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title(title)
    return fig

def correlation(df, columns):
    correl = df[columns].corr()
    return correl

def boxplot_numerical_cols(df, column):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter([1, 2, 3], [1, 2, 3])
    sns.boxplot(x=df[column], color='skyblue')
    plt.title(f'Box Plot de la variable {column}')
    plt.xlabel(column)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    return fig

def calculate_outliers(df, numerical_cols):
    outlier_info = {}
    for column in numerical_cols:  
        Q1 = df[column].quantile(0.15)  
        Q3 = df[column].quantile(0.85)  
        IQR = Q3 - Q1  # Étendue interquartile
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] > upper_bound)]
        outlier_percentage = (len(outliers) / len(df)) * 100
        outlier_info[column] = {
            'Upper Bound': upper_bound,
            'Outlier Count': len(outliers),
            'Outlier Percentage (%)': outlier_percentage
        }
    outlier_summary = pd.DataFrame(outlier_info).T
    return outlier_summary

def remove_outliers(df, columns):
    cleaned_df = df.copy()
    for col in columns:
        Q1 = df[col].quantile(0.15)
        Q3 = df[col].quantile(0.85)
        IQR = Q3 - Q1  # Étendue interquartile
        upper_bound = Q3 + 1.5 * IQR
        cleaned_df = cleaned_df[cleaned_df[col] <= upper_bound]
    return cleaned_df

def calculate_quartile(df, column, percentile):
    column_quartile = df[column].quantile(percentile)
    return column_quartile

def separate_bad_good_ratings(df, noteseuil, column):
    bad_ratings = df[df[column] <= noteseuil]
    good_ratings = df[df[column] > noteseuil]
    return bad_ratings, good_ratings

def plot_bad_ratings_distributions(bad_ratings, good_ratings):
    fig, ax = plt.subplots(2, 3, figsize=(8, 5))
    bad_ratings['minutes'].hist(ax=ax[0, 0])
    ax[0, 0].set_title('Distribution de Preparation Time')
    ax[0, 0].set_xlabel('Minutes')
    bad_ratings['n_steps'].hist(ax=ax[0,1])
    ax[0,1].set_title('Distribution de n_steps')
    ax[0,1].set_xlabel('Steps')
    bad_ratings['n_ingredients'].hist(ax=ax[0,2])
    ax[0,2].set_title('Distribution de n_ingredients')
    ax[0,2].set_xlabel('Ingredients')

    good_ratings['minutes'].hist(ax=ax[1,0])
    ax[1,0].set_title('Distribution de Preparation Time')
    ax[1,0].set_xlabel('Minutes')
    good_ratings['n_steps'].hist(ax=ax[1,1])
    ax[1,1].set_title('Distribution de n_steps')
    ax[1,1].set_xlabel('Steps')
    good_ratings['n_ingredients'].hist(ax=ax[1,2])
    ax[1,2].set_title('Distribution de n_ingredients')
    ax[1,2].set_xlabel('Ingredients')
    plt.tight_layout()
    return fig

def saisonnalite(df):
    # Grouper par jour de la semaine pour compter les soumissions
    day_of_week_bad = df['day_of_week'].value_counts().sort_index()
    # Grouper par mois pour compter les soumissions
    month_bad = df['month'].value_counts().sort_index()
    # Créer les barplots
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    # Barplot pour le nombre de soumissions selon le jour de la semaine
    ax[0].bar(day_of_week_bad.index, day_of_week_bad.values, color='lightcoral')
    ax[0].set_title('Nombre de soumissions par jour de la semaine', fontsize=14)
    ax[0].set_xlabel('Jour de la semaine')
    ax[0].set_ylabel('Nombre de soumissions')
    # Barplot pour le nombre de soumissions selon le mois de l'année
    ax[1].bar(month_bad.index, month_bad.values, color='lightblue')
    ax[1].set_title('Nombre de soumissions par mois', fontsize=14)
    ax[1].set_xlabel('Mois')
    ax[1].set_ylabel('Nombre de soumissions')
    # Ajuster l'affichage des graphiques
    plt.tight_layout()
    return fig

def boxplot_df(df):
    fig=plt.figure(figsize=(10, 6))
    plt.boxplot(df, labels=['Recettes bien notées', 'Recettes mal notées'])
    plt.title('Comparaison du temps de préparation')
    plt.ylabel('Minutes')
    plt.grid(True)
    plt.figure(figsize=(10, 6))
    return fig

def rating_distribution(df, variable, rating_var, low_threshold, mean_range, high_threshold, bins=[float('-inf'), 2, 3, 4, float('inf')], labels=['Less than 2', '2 to 3', '3 to 4', '4 to 5']):
    """
    Calcule la distribution des notes pour une variable donnée et retourne un graphique en barres empilées.

    Args:
    - df (DataFrame): DataFrame contenant les données.
    - variable (str): Nom de la variable à analyser.
    - rating_var (str): Nom de la variable de notation.
    - low_threshold (float): Seuil inférieur pour les catégories basses.
    - mean_range (tuple): Intervalle pour les catégories moyennes.
    - high_threshold (float): Seuil supérieur pour les catégories élevées.
    - bins (list): Bins pour les catégories de notes (default: [float('-inf'), 2, 3, 4, float('inf')]).
    - labels (list): Labels des bins (default: ['Less than 2', '2 to 3', '3 to 4', '4 to 5']).
    """
    
    def calculate_percentage(subset, total):
        subset = subset.copy()  # Créer une copie pour éviter SettingWithCopyWarning
        subset.loc[:, 'rating_category'] = pd.cut(subset[rating_var], bins=bins, labels=labels, right=True)
        category_counts = subset['rating_category'].value_counts().sort_index()
        return (category_counts / total) * 100
    # Catégories élevées (>= high_threshold)
    high_recipes = df[df[variable] > high_threshold]
    total_high = high_recipes.shape[0]
    category_percentages_high = calculate_percentage(high_recipes, total_high)
    # Catégories moyennes (entre mean_range[0] et mean_range[1])
    mean_recipes = df[(df[variable] >= mean_range[0]) & (df[variable] <= mean_range[1])]
    total_mean = mean_recipes.shape[0]
    category_percentages_mean = calculate_percentage(mean_recipes, total_mean)
    # Categories basses (< low_threshold)
    low_recipes = df[df[variable] < low_threshold]
    total_low = low_recipes.shape[0]
    category_percentages_low = calculate_percentage(low_recipes, total_low)
    # Combiner les résultats dans un DataFrame
    comparison_df = pd.DataFrame({
        'Catégorie élevée': category_percentages_high,
        'Catégorie moyenne': category_percentages_mean,
        'Catégorie basse': category_percentages_low
    })
    # Visualisation sous forme de Stacked Bar Chart
    stacked_df = pd.DataFrame({
        'Less than 2': [category_percentages_high.get('Less than 2', 0), category_percentages_mean.get('Less than 2', 0), category_percentages_low.get('Less than 2', 0)],
        '2 to 3': [category_percentages_high.get('2 to 3', 0), category_percentages_mean.get('2 to 3', 0), category_percentages_low.get('2 to 3', 0)],
        '3 to 4': [category_percentages_high.get('3 to 4', 0), category_percentages_mean.get('3 to 4', 0), category_percentages_low.get('3 to 4', 0)],
        '4 to 5': [category_percentages_high.get('4 to 5', 0), category_percentages_mean.get('4 to 5', 0), category_percentages_low.get('4 to 5', 0)]
    }, index=['Catégorie élevée', 'Catégorie moyenne', 'Catégorie basse'])
    # Plot 
    fig, ax = plt.subplots(figsize=(10, 6))
    stacked_df.plot(kind='bar', stacked=True, ax=ax, color=['red', 'orange', 'yellow', 'green'])
    ax.set_title(f'Stacked Distribution de la moyenne par rapport à la variable {variable}')
    ax.set_xlabel(f'{variable.capitalize()} Catégorie')
    ax.set_ylabel('Pourcentage de recettes (%)')
    return fig, comparison_df

def OLS_regression(X, y):
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model

def preprocess_text(text):
    text = text.lower() # Convertir en minuscule
    text = re.sub(r'[^\w\s]', '', text) # Retirer la punctuation et les caractères speciaux
    text = re.sub(r'\d+', '', text)  # Retirer les chiffres
    words = text.split() # Enlever les "stop words" (d'après liste fournie par sklearn)
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    return ' '.join(words)

def get_most_common_words(df) :
    all_tags_text = ' '.join(df.dropna())
    tag_words = all_tags_text.split()
    tag_word_counts = Counter(tag_words).most_common(100)
    return tag_word_counts

def extractWordFromTUpple(tup):
    return set(word for word, count in tup)

def uniqueTags(list1, list2):
    return (list1-list2)

def time_per_step(df, column1, column2):
    # Calculer le temps par étape
    df['minute_step'] = df[column1] / df[column2]
    df['minute_step_aberation'] = pd.cut(df['minute_step'], bins=[0, 2, 40, 150, float('inf')], labels=['quick', 'medium', 'long', 'very long'])
    # Calculer les pourcentages de ratings par minute_step
    rating_counts_min_step = pd.crosstab(df['minute_step_aberation'], df['rating'], normalize='index') * 100
    # Tracer le graphique
    ax = rating_counts_min_step.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
    ax.set_title('Pourcentage de ratings par niveau de minute par étape')
    ax.set_xlabel('Minute par étape')
    ax.set_ylabel('Pourcentage (%)')
    ax.legend(title='Rating', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig=ax.figure
    return fig

def rating_isContributor(df, column1) :
    fig=plt.figure(figsize=(8, 5))
    sns.countplot(x=column1, data=df, palette='pastel', hue=column1, legend=False)
    plt.title('Distribution de la variable is_contributor', fontsize=16)
    plt.xlabel('is_contributor', fontsize=14)
    plt.ylabel('Nombre d’utilisateurs', fontsize=14)
    plt.xticks([0, 1], labels=['Non-Contributeur', 'Contributeur'], fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    return fig

def plot_distributionIsContributor(df, X, Y):
    fig = plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[X], y=df[Y], hue=df[X], palette='pastel', legend=False)
    plt.title('Distribution des notes moyennes par type d’utilisateur', fontsize=16)
    plt.xlabel(X, fontsize=14)
    plt.ylabel(Y, fontsize=14)
    plt.xticks([0, 1], labels=['Non-Contributeur', 'Contributeur'], fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    return fig

def create_data_part1():
    data1 = load_csv("Pretraitement/recipe_mark.csv")
    data2 = append_csv(
                "Pretraitement/recipe_cleaned_part_1.csv",
                "Pretraitement/recipe_cleaned_part_2.csv",
                "Pretraitement/recipe_cleaned_part_3.csv",
                "Pretraitement/recipe_cleaned_part_4.csv",
                "Pretraitement/recipe_cleaned_part_5.csv")
    df = merged_data(data2, data1, "id", "recipe_id", "left") # Jointure entre data1 et data2
    drop_columns(df, ['recipe_id', 'nutrition', 'steps']) # Supprimer les colonnes en double
    df.columns = ['name', 'recipe_id', 'minutes', 'contributor_id', 'submitted', 'tags', 'n_steps', 
                    'description', 'ingredients', 'n_ingredients', 'calories', 'total_fat', 'sugar', 
                    'sodium','protein', 'saturated_fat', 'carbohydrates', 'year', 'month', 'day', 
                    'day_of_week', 'nb_user', 'note_moyenne','note_mediane', 'note_q1', 'note_q2', 
                    'note_q3', 'note_q4', 'note_max', 'note_min', 'nb_note_lt_5', 'nb_note_eq_5'] # Renommer les colonnes
    col_to_clean = ['minutes', 'n_steps', 'n_ingredients', 'calories', 'total_fat', 'sugar', 'sodium', 'protein', 'saturated_fat', 'carbohydrates']
    df_cleaned=remove_outliers(df, col_to_clean)
    data1 = None # Libérer la mémoire
    data2 = None # Libérer la mémoire
    df = None # Libérer la mémoire
    return df_cleaned

def create_data_part2():
    data2 = append_csv(
                    "Pretraitement/recipe_cleaned_part_1.csv",
                    "Pretraitement/recipe_cleaned_part_2.csv",
                    "Pretraitement/recipe_cleaned_part_3.csv",
                    "Pretraitement/recipe_cleaned_part_4.csv",
                    "Pretraitement/recipe_cleaned_part_5.csv")
    data3 = append_csv(
                    "Pretraitement/RAW_interactions_part_1.csv",
                    "Pretraitement/RAW_interactions_part_2.csv",
                    "Pretraitement/RAW_interactions_part_3.csv",
                    "Pretraitement/RAW_interactions_part_4.csv",
                    "Pretraitement/RAW_interactions_part_5.csv")
    user_analysis = merged_data(data3, data2, "recipe_id", "id", "left") # Jointure entre data2 et data3
    data2 = None # Libérer la mémoire
    data3 = None # Libérer la mémoire
    dropNa(user_analysis, ['name']) # 34 notes ne correspondent à aucune recette. Ce sont les outliers qu'on a sorti du dataset recipe lors de la première analyse. Nous allons les drop.
    fillNa(user_analysis, 'review', 'missing') # Remplacer les valeurs manquantes par 'missing'
    drop_columns(user_analysis, ['name', 'id','nutrition','steps', 'saturated fat (%)']) # Nous ne gardons que les colonnes utiles à l'analyse et non répétitive
    id_columns = ['recipe_id', 'user_id', 'contributor_id','year', 'month', 'day']
    for col in id_columns:
        user_analysis[col] = user_analysis[col].astype('object')
    user_analysis.columns = ['user_id', 'recipe_id', 'date', 'rating', 'review', 'minutes',
            'contributor_id', 'submitted', 'tags', 'n_steps', 'description',
            'ingredients', 'n_ingredients', 'calories', 'total_fat',
            'sugar', 'sodium', 'protein', 'carbohydrates', 'year',
            'month', 'day', 'day_of_week'] # On renomme les colonnes
    # Créer la variable binaire cible 'binary_rating' en fonction de la note
    # Mauvaise note (<=4) sera codée par 0, et bonne note (>4) par 1
    user_analysis['binary_rating'] = user_analysis['rating'].apply(lambda x: 0 if x <= 4 else 1)
    numerical_col = user_analysis.select_dtypes(include=['int64', 'float64']).columns
    col_to_clean = ['minutes', 'n_steps', 'n_ingredients', 'calories', 'total_fat', 'sugar','sodium', 'protein', 'carbohydrates']
    user_analysis_cleaned = remove_outliers(user_analysis, col_to_clean) # Supprimer les outliers
    user_analysis = None # Libérer la mémoire
    return user_analysis_cleaned

def create_dfuser_profiles(df) :
    # Regrouper par utilisateur pour calculer les métriques
    user_profiles = df.groupby('user_id').agg(
        num_recipes_rated=('rating', 'count'),       # Nombre de recettes notées
        mean_rating=('rating', 'mean'),             # Moyenne des notes
        median_rating=('rating', 'median'),         # Médiane des notes
        min_rating=('rating', 'min'),               # Note minimale
        max_rating=('rating', 'max'),               # Note maximale
        var_rating=('rating', 'var')                # Variance des notes
    ).reset_index()
    # Ajouter une colonne booléenne indiquant si l'utilisateur est contributeur
    user_profiles['is_contributor'] = user_profiles['user_id'].isin(df['contributor_id'])
    # Remplacer les NaN dans la colonne variance par 0 (au cas où un utilisateur n'a noté qu'une recette)
    user_profiles['var_rating'] = user_profiles['var_rating'].fillna(0)
    return user_profiles

















########################################################""

def main():
    recipe_path = "Pretraitement/recipe_cleaned.csv"
    stat_user_path = "Pretraitement/recipe_mark.csv"
    
    merged_df = load_data(recipe_path, stat_user_path)
    check_duplicates(merged_df)
    
    columns_to_drop = ['recipe_id', 'nutrition', 'steps']
    drop_columns(merged_df, columns_to_drop)
    
    new_column_names = ['name', 'recipe_id', 'minutes', 'contributor_id', 'submitted', 'tags',
                        'n_steps', 'description', 'ingredients', 'n_ingredients', 'calories',
                        'total_fat', 'sugar', 'sodium', 'protein', 'saturated_fat', 'carbohydrates',
                        'year', 'month', 'day', 'day_of_week', 'nb_user', 'note_moyenne', 'note_mediane',
                        'note_q1', 'note_q2', 'note_q3', 'note_q4', 'note_max', 'note_min', 'nb_note_lt_5', 'nb_note_eq_5']
    rename_columns(merged_df, new_column_names)
    
    plot_distribution(merged_df, ['note_moyenne', 'note_mediane'], ['Distribution de la moyenne', 'Distribution de la médiane'])
    
    correlation_columns = ['note_moyenne', 'minutes', 'n_steps', 'n_ingredients', 'calories', 'total_fat', 
                           'sugar', 'sodium', 'protein', 'saturated_fat', 'carbohydrates', 'nb_user']
    plot_correlation_matrix(merged_df, correlation_columns, 'Matrice de correlation des variables avec la moyenne')
    
    numerical_cols = merged_df.select_dtypes(include=['int64', 'float64']).columns
    calculate_outliers(merged_df, numerical_cols)
    
    col_to_clean = ['minutes', 'n_steps', 'n_ingredients', 'calories', 'total_fat', 'sugar',
                    'sodium', 'protein', 'saturated_fat', 'carbohydrates']
    cleaned_df = remove_outliers(merged_df, col_to_clean)
    
    print(f"Taille initiale du DataFrame : {merged_df.shape}")
    print(f"Taille après suppression des outliers : {cleaned_df.shape}")
    
    plot_correlation_matrix(cleaned_df, correlation_columns, 'Matrice de correlation des variables avec la moyenne (après suppression des outliers)')
    
    mean_third_quartile, median_third_quartile = calculate_third_quartiles(cleaned_df)
    
    # Afficher les résultats
    print("3e Quartile pour la moyenne:", mean_third_quartile)
    print("3e Quartile pour la médiane:", median_third_quartile)
    print('Nombre de recettes avec une moyenne inférieure à 4 :')
    print(cleaned_df[cleaned_df['note_moyenne'] <= 4.0].shape[0])
    print('Nombre de recettes avec une médiane inférieure à 4 :')
    print(cleaned_df[cleaned_df['note_mediane'] <= 4.0].shape[0])
    
    plot_bad_ratings_distributions(cleaned_df)

if __name__ == "__main__":
    main()