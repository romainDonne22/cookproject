import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
#import re
#from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
#from collections import Counter
#from textblob import TextBlob
#from wordcloud import WordCloud
#from sklearn.preprocessing import StandardScaler


def load_data(fichier):
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
    
def merged_data(data1, data2):
    try:
        merged_df = pd.merge(data2, data1, left_on="id", right_on="recipe_id", how="left")
        return merged_df
    except Exception as e:
        print(f"Failed to load data: {e}")
        return pd.DataFrame()

def check_duplicates(df):
    num_duplicates = df.duplicated().sum()
    return num_duplicates

def drop_columns(df, columns_to_drop):
    df.drop(columns_to_drop, axis=1, inplace=True)

def rename_columns(df, new_column_names):
    df.columns = new_column_names

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

def calculate_third_quartiles(df, column):
    column_third_quartile = df[column].quantile(0.25)
    return column_third_quartile

def separate_bad_good_ratings(df, noteseuil):
    bad_ratings = df[df['note_moyenne'] <= noteseuil]
    good_ratings = df[df['note_moyenne'] > noteseuil]
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
    ax[1,0].set_title('Distribution de Preparation Time XSDDS')
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
    plt.show()
    plt.figure(figsize=(10, 6))
    return fig


def rating_distribution(df, variable, rating_var, low_threshold, mean_range, high_threshold,
                        bins=[float('-inf'), 2, 3, 4, 5], labels=['Less than 2', '2 to 3', '3 to 4', '4 to 5']):
    
    """
    Calculer la distribution des moyennes pour chaque catégorie d'une variable.
    
    Paramètres :
    - df (pd.DataFrame): input DataFrame.
    - variable (str): le nom de la colonne à filtrer (e.g., 'n_steps').
    - rating_var (str): le nom de la colonne pour laquelle on veut observer les proportions (e.g., 'note_moyenne')
    - low_threshold (float): seuil des valeurs considérées basses
    - mean_range (tuple): échelle (min, max) pour les valeurs considérées dans la moyenne
    - high_threshold (float): seuil des valeurs considérées élevées
    - bins (list): liste des bins pour la moyenne (default: [float('-inf'), 2, 3, 4, 5]).
    - labels (list): Labels des bins (default: ['Less than 2', '2 to 3', '3 to 4', '4 to 5']).
    
    """
    
    def calculate_percentage(subset, total):
        subset['rating_category'] = pd.cut(subset[rating_var], bins=bins, labels=labels, right=True)
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