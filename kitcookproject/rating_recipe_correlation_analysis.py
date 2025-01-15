import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from collections import Counter
from wordcloud import WordCloud


def init_data_part1():
    """
    Initialize and clean the first part of the data.

    Returns:
        DataFrame: Cleaned DataFrame for the first part of the data.
    """
    df_cleaned = create_data_part1()
    return df_cleaned

def init_data_part2():
    """
    Initialize and clean the second part of the data.

    Returns:
        DataFrame: Cleaned DataFrame for the second part of the data.
    """
    user_analysis_cleaned = create_data_part2()
    return user_analysis_cleaned

def load_csv(fichier):
    """
    Load a CSV file into a DataFrame.

    Args:
        fichier (str): Path to the CSV file.

    Returns:
        DataFrame: Loaded DataFrame.
    """
    try:
        data = pd.read_csv(fichier)
        return data
    except Exception as e:
        print(f"Failed to load data: {e}")
        return pd.DataFrame()


def append_csv(*files):
    """
    Append multiple CSV files into a single DataFrame.

    Args:
        *files (str): Paths to the CSV files.

    Returns:
        DataFrame: Concatenated DataFrame.
    """
    df_list = []
    for file in files:
        df = pd.read_csv(file)
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)


def merged_data(data1, data2, LO, RO, H):
    """
    Merge two DataFrames.

    Args:
        data1 (DataFrame): First DataFrame.
        data2 (DataFrame): Second DataFrame.
        LO (str): Column name in the first DataFrame to join on.
        RO (str): Column name in the second DataFrame to join on.
        H (str): Type of join to perform.

    Returns:
        DataFrame: Merged DataFrame.
    """
    try:
        merged_df = pd.merge(data1, data2, left_on=LO, right_on=RO, how=H)
        return merged_df
    except Exception as e:
        print(f"Failed to load data: {e}")
        return pd.DataFrame()


def check_duplicates(df):
    """
    Check for duplicate rows in a DataFrame.

    Args:
        df (DataFrame): DataFrame to check for duplicates.

    Returns:
        int: Number of duplicate rows.
    """
    num_duplicates = df.duplicated().sum()
    return num_duplicates


def drop_columns(df, columns_to_drop):
    """
    Drop specified columns from a DataFrame.

    Args:
        df (DataFrame): DataFrame to drop columns from.
        columns_to_drop (list): List of column names to drop.
    """
    df.drop(columns_to_drop, axis=1, inplace=True)


def dropNa(df, columns_to_drop):
    """
    Drop rows with missing values in specified columns.

    Args:
        df (DataFrame): DataFrame to drop rows from.
        columns_to_drop (list): List of column names to check for missing values.
    """
    df.dropna(subset=columns_to_drop, inplace=True)


def fillNa(df, column_to_fill, value):
    """
    Fill missing values in a specified column with a given value.

    Args:
        df (DataFrame): DataFrame to fill missing values in.
        column_to_fill (str): Column name to fill missing values in.
        value: Value to fill missing values with.
    """
    df[column_to_fill].fillna(value, inplace=True)


def dflog(df, column):
    """
    Apply log transformation to a specified column in a DataFrame.

    Args:
        df (DataFrame): DataFrame to apply log transformation to.
        column (str): Column name to apply log transformation to.

    Returns:
        Series: Transformed column.
    """
    df['newcolumn'] = np.log1p(df[column])
    return df['newcolumn']


def plot_distribution(df, column, title):
    """
    Plot the distribution of a specified column in a DataFrame.

    Args:
        df (DataFrame): DataFrame to plot.
        column (str): Column name to plot.
        title (str): Title of the plot.

    Returns:
        Figure: Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter([1, 2, 3], [1, 2, 3])
    plt.hist(df[column], bins=20)
    plt.title(title)
    plt.xlabel(title)
    plt.ylabel('Fréquence')
    plt.tight_layout()
    return fig


def plot_correlation_matrix(df, columns, title):
    """
    Plot the correlation matrix of specified columns in a DataFrame.

    Args:
        df (DataFrame): DataFrame to plot.
        columns (list): List of column names to include in the correlation matrix.
        title (str): Title of the plot.

    Returns:
        Figure: Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter([1, 2, 3], [1, 2, 3])
    correlation = df[columns].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title(title)
    return fig


def correlation(df, columns):
    """
    Calculate the correlation matrix of specified columns in a DataFrame.

    Args:
        df (DataFrame): DataFrame to calculate correlation matrix for.
        columns (list): List of column names to include in the correlation matrix.

    Returns:
        DataFrame: Correlation matrix.
    """
    correl = df[columns].corr()
    return correl


def boxplot_numerical_cols(df, column):
    """
    Plot a boxplot for a specified numerical column in a DataFrame.

    Args:
        df (DataFrame): DataFrame to plot.
        column (str): Column name to plot.

    Returns:
        Figure: Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter([1, 2, 3], [1, 2, 3])
    sns.boxplot(x=df[column], color='skyblue')
    plt.title(f'Box Plot de la variable {column}')
    plt.xlabel(column)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    return fig


def calculate_outliers(df, numerical_cols):
    """
    Calculate outliers for specified numerical columns in a DataFrame.

    Args:
        df (DataFrame): DataFrame to calculate outliers for.
        numerical_cols (list): List of numerical column names to check for outliers.

    Returns:
        DataFrame: Summary of outliers.
    """
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
    """
    Remove outliers from specified columns in a DataFrame.

    Args:
        df (DataFrame): DataFrame to remove outliers from.
        columns (list): List of column names to check for outliers.

    Returns:
        DataFrame: DataFrame with outliers removed.
    """
    cleaned_df = df.copy()
    for col in columns:
        Q1 = df[col].quantile(0.15)
        Q3 = df[col].quantile(0.85)
        IQR = Q3 - Q1  # Étendue interquartile
        upper_bound = Q3 + 1.5 * IQR
        cleaned_df = cleaned_df[cleaned_df[col] <= upper_bound]
    return cleaned_df


def calculate_quartile(df, column, percentile):
    """
    Calculate the quartile for a specified column in a DataFrame.

    Args:
        df (DataFrame): DataFrame to calculate quartile for.
        column (str): Column name to calculate quartile for.
        percentile (float): Percentile to calculate.

    Returns:
        float: Calculated quartile.
    """
    column_quartile = df[column].quantile(percentile)
    return column_quartile


def separate_bad_good_ratings(df, noteseuil, column):
    """
    Separate DataFrame into bad and good ratings based on a threshold.

    Args:
        df (DataFrame): DataFrame to separate.
        noteseuil (float): Threshold for separating ratings.
        column (str): Column name to base the separation on.

    Returns:
        tuple: DataFrames of bad and good ratings.
    """
    bad_ratings = df[df[column] <= noteseuil]
    good_ratings = df[df[column] > noteseuil]
    return bad_ratings, good_ratings


def plot_bad_ratings_distributions(bad_ratings, good_ratings):
    """
    Plot distributions of preparation time, number of steps, and ingredients for bad and good ratings.

    Args:
        bad_ratings (DataFrame): DataFrame of bad ratings.
        good_ratings (DataFrame): DataFrame of good ratings.

    Returns:
        Figure: Matplotlib figure object.
    """
    fig, ax = plt.subplots(2, 3, figsize=(8, 5))
    bad_ratings['minutes'].hist(ax=ax[0, 0])
    ax[0, 0].set_title('Distribution de Preparation Time')
    ax[0, 0].set_xlabel('Minutes')
    bad_ratings['n_steps'].hist(ax=ax[0, 1])
    ax[0, 1].set_title('Distribution de n_steps')
    ax[0, 1].set_xlabel('Steps')
    bad_ratings['n_ingredients'].hist(ax=ax[0, 2])
    ax[0, 2].set_title('Distribution de n_ingredients')
    ax[0, 2].set_xlabel('Ingredients')

    good_ratings['minutes'].hist(ax=ax[1, 0])
    ax[1, 0].set_title('Distribution de Preparation Time')
    ax[1, 0].set_xlabel('Minutes')
    good_ratings['n_steps'].hist(ax=ax[1, 1])
    ax[1, 1].set_title('Distribution de n_steps')
    ax[1, 1].set_xlabel('Steps')
    good_ratings['n_ingredients'].hist(ax=ax[1, 2])
    ax[1, 2].set_title('Distribution de n_ingredients')
    ax[1, 2].set_xlabel('Ingredients')
    plt.tight_layout()
    return fig


def saisonnalite(df):
    """
    Plot the number of submissions by day of the week and by month.

    Args:
        df (DataFrame): DataFrame to plot.

    Returns:
        Figure: Matplotlib figure object.
    """
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
    """
    Plot a boxplot comparing preparation times for well-rated and poorly-rated recipes.

    Args:
        df (DataFrame): DataFrame to plot.

    Returns:
        Figure: Matplotlib figure object.
    """
    fig = plt.figure(figsize=(10, 6))
    plt.boxplot(df, labels=['Recettes bien notées', 'Recettes mal notées'])
    # plt.title('Comparaison du temps de préparation')
    plt.ylabel('Minutes')
    plt.grid(True)
    return fig

def rating_distribution(df, variable, rating_var, low_threshold, mean_range, high_threshold, bins=[float('-inf'), 2, 3, 4, float('inf')], labels=['Less than 2', '2 to 3', '3 to 4', '4 to 5']):
    """
    Calculate the distribution of ratings for a given variable and return a stacked bar chart.

    Args:
    - df (DataFrame): DataFrame containing the data.
    - variable (str): Name of the variable to analyze.
    - rating_var (str): Name of the rating variable.
    - low_threshold (float): Lower threshold for low categories.
    - mean_range (tuple): Range for medium categories.
    - high_threshold (float): Upper threshold for high categories.
    - bins (list): Bins for rating categories (default: [float('-inf'), 2, 3, 4, float('inf')]).
    - labels (list): Labels for the bins (default: ['Less than 2', '2 to 3', '3 to 4', '4 to 5']).
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
    """
    Perform an Ordinary Least Squares (OLS) regression.

    Args:
        X (DataFrame): Independent variables.
        y (Series): Dependent variable.

    Returns:
        RegressionResults: Fitted OLS model.
    """
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model


def preprocess_text(text):
    """
    Preprocess text by converting to lowercase, removing punctuation, special characters, and stop words.

    Args:
        text (str): Text to preprocess.

    Returns:
        str: Preprocessed text.
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation and special characters
    text = re.sub(r'\d+', '', text)  # Remove digits
    words = text.split()  # Split into words
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]  # Remove stop words
    return ' '.join(words)


def get_most_common_words(df):
    """
    Get the 100 most common words from a DataFrame column.

    Args:
        df (DataFrame): DataFrame containing text data.

    Returns:
        list: List of tuples with the 100 most common words and their counts.
    """
    all_tags_text = ' '.join(df.dropna())
    tag_words = all_tags_text.split()
    tag_word_counts = Counter(tag_words).most_common(100)
    return tag_word_counts


def extractWordFromTUpple(tup):
    """
    Extract words from a list of tuples.

    Args:
        tup (list): List of tuples containing words and their counts.

    Returns:
        set: Set of words.
    """
    return set(word for word, count in tup)


def uniqueTags(list1, list2):
    """
    Get unique tags from two sets.

    Args:
        list1 (set): First set of tags.
        list2 (set): Second set of tags.

    Returns:
        set: Unique tags in the first set that are not in the second set.
    """
    return list1 - list2


def time_per_step(df, column1, column2):
    """
    Calculate the time per step and plot the percentage of ratings by time per step.

    Args:
        df (DataFrame): DataFrame containing the data.
        column1 (str): Column name for total time.
        column2 (str): Column name for number of steps.

    Returns:
        Figure: Matplotlib figure object.
    """
    df['minute_step'] = df[column1] / df[column2]
    df['minute_step_aberation'] = pd.cut(
        df['minute_step'],
        bins=[0, 2, 40, 150, float('inf')],
        labels=['quick', 'medium', 'long', 'very long']
    )
    rating_counts_min_step = pd.crosstab(
        df['minute_step_aberation'], df['rating'], normalize='index'
    ) * 100
    ax = rating_counts_min_step.plot(
        kind='bar', stacked=True, figsize=(10, 6), colormap='viridis'
    )
    ax.set_title('Pourcentage de ratings par niveau de minute par étape')
    ax.set_xlabel('Minute par étape')
    ax.set_ylabel('Pourcentage (%)')
    ax.legend(title='Rating', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig = ax.figure
    return fig


def rating_isContributor(df, column1):
    """
    Plot the distribution of the is_contributor variable.

    Args:
        df (DataFrame): DataFrame containing the data.
        column1 (str): Column name for the is_contributor variable.

    Returns:
        Figure: Matplotlib figure object.
    """
    fig = plt.figure(figsize=(8, 5))
    sns.countplot(x=column1, data=df, palette='pastel', hue=column1, legend=False)
    plt.title('Distribution de la variable is_contributor', fontsize=16)
    plt.xlabel('is_contributor', fontsize=14)
    plt.ylabel('Nombre d’utilisateurs', fontsize=14)
    plt.xticks([0, 1], labels=['Non-Contributeur', 'Contributeur'], fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    return fig


def plot_distributionIsContributor(df, X, Y):
    """
    Plot the distribution of ratings by contributor type.

    Args:
        df (DataFrame): DataFrame containing the data.
        X (str): Column name for the contributor type.
        Y (str): Column name for the ratings.

    Returns:
        Figure: Matplotlib figure object.
    """
    fig = plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[X], y=df[Y], hue=df[X], palette='pastel', legend=False)
    plt.title('Distribution des notes moyennes par type d’utilisateur', fontsize=16)
    plt.xlabel(X, fontsize=14)
    plt.ylabel(Y, fontsize=14)
    plt.xticks([0, 1], labels=['Non-Contributeur', 'Contributeur'], fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    return fig

def create_data_part1():
    """
    Create and clean the first part of the data.

    This function loads and merges multiple CSV files, drops unnecessary columns,
    renames columns, removes outliers, and returns the cleaned DataFrame.

    Returns:
        DataFrame: Cleaned DataFrame for the first part of the data.
    """
    data1 = load_csv("Pretraitement/recipe_mark.csv")
    data2 = append_csv(
        "Pretraitement/recipe_cleaned_part_1.csv",
        "Pretraitement/recipe_cleaned_part_2.csv",
        "Pretraitement/recipe_cleaned_part_3.csv",
        "Pretraitement/recipe_cleaned_part_4.csv",
        "Pretraitement/recipe_cleaned_part_5.csv"
    )
    df = merged_data(data2, data1, "id", "recipe_id", "left")  # Join data1 and data2
    drop_columns(df, ['recipe_id', 'nutrition', 'steps'])  # Drop duplicate columns
    df.columns = [
        'name', 'recipe_id', 'minutes', 'contributor_id', 'submitted', 'tags', 'n_steps',
        'description', 'ingredients', 'n_ingredients', 'calories', 'total_fat', 'sugar',
        'sodium', 'protein', 'saturated_fat', 'carbohydrates', 'year', 'month', 'day',
        'day_of_week', 'nb_user', 'note_moyenne', 'note_mediane', 'note_q1', 'note_q2',
        'note_q3', 'note_q4', 'note_max', 'note_min', 'nb_note_lt_5', 'nb_note_eq_5'
    ]  # Rename columns
    col_to_clean = [
        'minutes', 'n_steps', 'n_ingredients', 'calories', 'total_fat', 'sugar', 'sodium',
        'protein', 'saturated_fat', 'carbohydrates'
    ]
    df_cleaned = remove_outliers(df, col_to_clean)
    data1 = None  # Free memory
    data2 = None  # Free memory
    df = None  # Free memory
    return df_cleaned


def create_data_part2():
    """
    Create and clean the second part of the data.

    This function loads and merges multiple CSV files, drops unnecessary columns,
    renames columns, removes outliers, and returns the cleaned DataFrame.

    Returns:
        DataFrame: Cleaned DataFrame for the second part of the data.
    """
    data2 = append_csv(
        "Pretraitement/recipe_cleaned_part_1.csv",
        "Pretraitement/recipe_cleaned_part_2.csv",
        "Pretraitement/recipe_cleaned_part_3.csv",
        "Pretraitement/recipe_cleaned_part_4.csv",
        "Pretraitement/recipe_cleaned_part_5.csv"
    )
    data3 = append_csv(
        "Pretraitement/RAW_interactions_part_1.csv",
        "Pretraitement/RAW_interactions_part_2.csv",
        "Pretraitement/RAW_interactions_part_3.csv",
        "Pretraitement/RAW_interactions_part_4.csv",
        "Pretraitement/RAW_interactions_part_5.csv"
    )
    user_analysis = merged_data(data3, data2, "recipe_id", "id", "left")  # Join data2 and data3
    data2 = None  # Free memory
    data3 = None  # Free memory
    dropNa(user_analysis, ['name'])  # Drop rows with missing 'name' values
    fillNa(user_analysis, 'review', 'missing')  # Fill missing 'review' values with 'missing'
    drop_columns(user_analysis, ['name', 'id', 'nutrition', 'steps', 'saturated fat (%)'])  # Drop unnecessary columns
    id_columns = ['recipe_id', 'user_id', 'contributor_id', 'year', 'month', 'day']
    for col in id_columns:
        user_analysis[col] = user_analysis[col].astype('object')
    user_analysis.columns = [
        'user_id', 'recipe_id', 'date', 'rating', 'review', 'minutes', 'contributor_id',
        'submitted', 'tags', 'n_steps', 'description', 'ingredients', 'n_ingredients',
        'calories', 'total_fat', 'sugar', 'sodium', 'protein', 'carbohydrates', 'year',
        'month', 'day', 'day_of_week'
    ]  # Rename columns
    # Create binary target variable 'binary_rating' based on rating
    # Bad rating (<=4) is coded as 0, and good rating (>4) as 1
    user_analysis['binary_rating'] = user_analysis['rating'].apply(lambda x: 0 if x <= 4 else 1)
    numerical_col = user_analysis.select_dtypes(include=['int64', 'float64']).columns
    col_to_clean = [
        'minutes', 'n_steps', 'n_ingredients', 'calories', 'total_fat', 'sugar', 'sodium',
        'protein', 'carbohydrates'
    ]
    user_analysis_cleaned = remove_outliers(user_analysis, col_to_clean)  # Remove outliers
    user_analysis = None  # Free memory
    return user_analysis_cleaned


def create_dfuser_profiles(df):
    """
    Create user profiles by aggregating metrics for each user.

    Args:
        df (DataFrame): DataFrame containing user data.

    Returns:
        DataFrame: DataFrame with user profiles and metrics.
    """
    user_profiles = df.groupby('user_id').agg(
        num_recipes_rated=('rating', 'count'),  # Number of recipes rated
        mean_rating=('rating', 'mean'),  # Average rating
        median_rating=('rating', 'median'),  # Median rating
        min_rating=('rating', 'min'),  # Minimum rating
        max_rating=('rating', 'max'),  # Maximum rating
        var_rating=('rating', 'var')  # Rating variance
    ).reset_index()
    # Add a boolean column indicating if the user is a contributor
    user_profiles['is_contributor'] = user_profiles['user_id'].isin(df['contributor_id'])
    # Replace NaN in the variance column with 0 (in case a user has rated only one recipe)
    user_profiles['var_rating'] = user_profiles['var_rating'].fillna(0)
    return user_profiles


########################################################""

def main():
    """
    Main function to analyze recipe data.

    This function initializes and cleans the data, performs various analyses,
    and prints the results. It includes distribution plots, correlation matrices,
    quartile calculations, and linear regression analyses.

    Returns:
        None
    """
    df_cleaned = init_data_part1()  # Load the first dataset
    user_analysis_cleaned = init_data_part2()  # Load the second dataset

    data1 = df_cleaned
    data2 = user_analysis_cleaned

    print(data1.head())
    print(data2.head())

    nb_doublon = check_duplicates(data1)
    print(nb_doublon)
    plot_distribution(data1, 'note_moyenne', 'Distribution de la moyenne')
    plt.show()
    plot_distribution(data1, 'note_mediane', 'Distribution de la médiane')
    plt.show()
    plot_distribution(data2, 'rating', 'Distribution de la moyenne')
    plt.show()

    plot_correlation_matrix(
        data1,
        ['note_moyenne', 'minutes', 'n_steps', 'n_ingredients', 'calories', 'total_fat',
         'sugar', 'sodium', 'protein', 'saturated_fat', 'carbohydrates', 'nb_user'],
        "Matrice de corrélation entre la moyenne des notes et les autres variables numériques"
    )
    plt.show()
    plot_correlation_matrix(
        data2,
        ['rating', 'minutes', 'n_steps', 'n_ingredients', 'calories', 'total_fat', 'sugar',
         'sodium', 'protein', 'carbohydrates', 'binary_rating'],
        "Matrice de corrélation entre les notes et les autres variables numériques"
    )
    plt.show()

    numerical_cols = data1.select_dtypes(include=['int64', 'float64']).columns
    for colonne in numerical_cols:
        boxplot_numerical_cols(data1, colonne)

    # Calculate quartiles
    mean_quartile = calculate_quartile(data1, 'note_moyenne', 0.25)
    print("3e Quartile pour la moyenne:", mean_quartile)
    mean_quartile = calculate_quartile(data1, 'note_mediane', 0.25)
    print("3e Quartile pour la médiane:", mean_quartile)
    # Number of bad ratings
    print(f"Nombre de recettes avec une moyenne inférieure à 4 : {data1[data1['note_moyenne'] <= 4.0].shape[0]}")
    print(f"Nombre de recettes avec une médiane inférieure à 4 : {data1[data1['note_mediane'] <= 4.0].shape[0]}")
    print("Nous nous concentrerons sur la moyenne qui nous permet d'augmenter l'échantillon de bad ratings. "
          "Compte tenu de la distribution de la moyenne, on peut considérer les 4 (et moins) comme des mauvaises notes.")
    # Separate bad and good ratings
    bad_ratings, good_ratings = separate_bad_good_ratings(data1, 4, 'note_moyenne')

    # Calculate quartiles
    mean_quartile = calculate_quartile(data2, 'rating', 0.25)
    print("3e Quartile pour la note:", mean_quartile)
    # Number of bad ratings
    print(f"Nombre de recettes avec une note inférieure à 4 : {data2[data2['rating'] <= 4.0].shape[0]}")
    print("Nous nous concentrerons sur la note qui nous permet d'augmenter l'échantillon de bad ratings. "
          "Compte tenu de la distribution de la note, on peut considérer les 4 (et moins) comme des mauvaises notes.")
    # Separate bad and good ratings
    bad_ratings, good_ratings = separate_bad_good_ratings(data2, 4, 'rating')

    # Filter recipes with a rating of 4 or less (from data2)
    print("Afin de comparer les recettes mal notées des bien notées, nous devons filtrer le dataframe sur les mauvaises notes (première ligne) et les bonnes notes (deuxième ligne). ")
    plot_bad_ratings_distributions(bad_ratings, good_ratings)
    plt.show()
    print("Pas de grosses variations à observer... Regardons maintenant si la saisonnalité / la période où la recette est postée a un impact :)")

    # Seasonality
    print("Saisonalié des recettes mal notées (en haut) et bien notées (en bas) : ")
    saisonnalite(bad_ratings)
    plt.show()
    saisonnalite(good_ratings)
    print("Nous n'observons pas d'impact de la saisonnalité du post entre bad et good ratings.")
    plt.show()

    data_minutes = [good_ratings['minutes'], bad_ratings['minutes']]
    boxplot_df(data_minutes)
    plt.show()
    data_steps = [good_ratings['n_steps'], bad_ratings['n_steps']]
    boxplot_df(data_steps)
    plt.show()
    data_ingred = [good_ratings['n_ingredients'], bad_ratings['n_ingredients']]
    boxplot_df(data_ingred)
    plt.show()
    print("Les recettes mal notées tendent à avoir des temps de préparation plus longs et un nombre d'étapes à suivre plus élevé. Rien à signalier sur le nombre d'ingrédients.")

    # Distribution of ratings by variable (minutes / n_steps / n_ingredients) in %:
    fig, comparison_minutes = rating_distribution(df=data2, variable='minutes', rating_var='rating',
                                                  low_threshold=15, mean_range=(30, 50), high_threshold=180)
    plt.show()
    print("Distribution de la note par rapport à la variable minutes en %:")
    print(comparison_minutes)
    fig, comparison_steps = rating_distribution(df=data2, variable='n_steps', rating_var='rating',
                                                low_threshold=3, mean_range=(8, 10), high_threshold=15)
    plt.show()
    print("Distribution de la note par rapport à la variable n_steps en %:")
    print(comparison_steps)
    fig, comparison_ingr = rating_distribution(df=data2, variable='n_ingredients', rating_var='rating',
                                               low_threshold=3, mean_range=(8, 10), high_threshold=15)
    plt.show()

    print("Distribution de la note par rapport à la variable n_ingredients en %:")
    print(comparison_ingr)
    print("Même analyse pour la variable nombre d'étapes : plus les recettes ont un nombre d'étapes élevé / sont complexes plus elles sont mal notées. "
          "A contrario les recettes avec moins de 3 étapes sont sensiblement mieux notées.")
    print("Le nombre d'ingrédients en revanche ne semble pas impacté la moyenne.")

    print("Réalisons une régression avec ces trois variables pour comprendre dans quelle mesure elles impactent la note et si cette hypothèse est statistiquement viable.")
    print("La matrice de corrélation en les variables 'minutes','n_steps','n_ingredients' est la suivante")
    columns_to_analyze = ['minutes', 'n_steps', 'n_ingredients']

    print(correlation(data2, columns_to_analyze))

    data = data1
    df1 = 1
    # Linear regression
    print("Régression linéaire entre les variables 'minutes','n_steps','n_ingredients' et la note moyenne : ")
    X = data[['minutes', 'n_steps']]
    if df1 == 1:
        y = data['note_moyenne']
    else:
        y = data['rating']
    model = OLS_regression(X, y)
    print(model.summary())
    print("ANALYSE :")
    print("R-Squared = O.OO1 -> seulement 0.1 pourcent de la variance dans les résultats est expliquée par les variables n_steps et minutes. "
          "C'est très bas, ces variables ne semblent pas avoir de pouvoir prédictif sur les ratings, même si on a pu détecter des tendances de comportements users.")
    print("Prob (F-Stat) = p-value est statistiquement signifiante (car < 0.05) -> au moins un estimateur a une relation linéaire avec note_moyenne. "
          "Cependant l'effet sera minime, comme le montre le résultat R-Squared")
    print("Coef minute : VERY small. p-value < 0.05 donc statistiquement signifiant mais son effet est quasi négligeable sur note_moyenne. "
          "Même constat pour n_steps même si l'effet est légèrement supérieur : une augmentation de 10 étapes va baisser la moyenne d'environ 0.025...")
    print("Les tests Omnibus / Prob(Omnibus) et Jarque-Bera (JB) / Prob(JB) nous permettent de voir que les résidus ne suivent probablement pas une distribution gaussienne, les conditions pour une OLS ne sont donc pas remplies.")
    print("--> il va falloir utiliser une log transformation pour s'approcher de variables gaussiennes.")

    # Linear regression with log transformation
    data['minutes_log'] = dflog(data, 'minutes')
    data['n_steps_log'] = dflog(data, 'n_steps')
    X = data[['minutes_log', 'n_steps_log']]
    if df1 == 1:
        y = data['note_moyenne']
    else:
        y = data['rating']
    model = OLS_regression(X, y)
    print(model.summary())
    print("ANALYSE :")
    print("En passant au log, on se rend compte que la variable minute a plus de poids sur la moyenne que le nombre d'étapes. "
          "Néanmoins bien que les variables minutes_log et n_steps_log soient statistiquement significatives (cf p value), leur contribution à la prédiction de la note moyenne est très faible.")
    print("En effet R2 est toujours extrêmement petit donc ces deux variables ont un impact minime sur la moyenne, qui ne permet pas d'expliquer les variations de la moyenne.")
    print("Il est probablement nécessaire d'explorer d'autres variables explicatives ou d'utiliser un modèle non linéaire pour mieux comprendre la note_moyenne.")

    # Page 4

    # Comparison of calories
    if df1 == 1:
        fig, comparison_calories = rating_distribution(df=data, variable='calories', rating_var='note_moyenne',
                                                       low_threshold=100, mean_range=(250, 350), high_threshold=1000)
    else:
        fig, comparison_calories = rating_distribution(df=data, variable='calories', rating_var='rating',
                                                       low_threshold=100, mean_range=(250, 350), high_threshold=1000)
    print("\nComparison of Rating Distribution in %:")
    plt.show()
    print("Distribution de la note par rapport à la variable calories en %:")
    print(comparison_calories)
    # Comparison of total_fat
    if df1 == 1:
        fig, comparison_total_fat = rating_distribution(df=data, variable='total_fat', rating_var='note_moyenne',
                                                        low_threshold=8, mean_range=(15, 25), high_threshold=100)
    else:
        fig, comparison_total_fat = rating_distribution(df=data, variable='total_fat', rating_var='rating',
                                                        low_threshold=8, mean_range=(15, 25), high_threshold=100)
    print("\nComparison of Rating Distribution in %:")
    plt.show()
    print("Distribution de la note par rapport à la variable total_fat en %:")
    print(comparison_total_fat)
    # Comparison of sugar
    if df1 == 1:
        fig, comparison_sugar = rating_distribution(df=data, variable='sugar', rating_var='note_moyenne',
                                                    low_threshold=8, mean_range=(15, 25), high_threshold=60)
    else:
        fig, comparison_sugar = rating_distribution(df=data, variable='sugar', rating_var='rating',
                                                    low_threshold=8, mean_range=(15, 25), high_threshold=60)
    print("\nComparison of Rating Distribution in %:")
    plt.show()
    print("Distribution de la note par rapport à la variable sugar en %:")
    print(comparison_sugar)
    # Comparison of protein
    if df1 == 1:
        fig, comparison_protein = rating_distribution(df=data, variable='protein', rating_var='note_moyenne',
                                                      low_threshold=8, mean_range=(15, 25), high_threshold=60)
    else:
        fig, comparison_protein = rating_distribution(df=data, variable='protein', rating_var='rating',
                                                      low_threshold=8, mean_range=(15, 25), high_threshold=60)
    print("\nComparison of Rating Distribution in %:")
    plt.show()
    print("Distribution de la note par rapport à la variable protein en %:")
    print(comparison_protein)
    # Conclusion
    print("Les variations sont trop faibles. Les contenus nutritionnels des recettes n'impactent pas la moyenne.")

    # Page 5

    if df1 == 1:
        # Calculate Q1, Q3, and IQR for nb_users
        Q1_nb_user = calculate_quartile(data, 'nb_user', 0.25)
        Q2_nb_user = calculate_quartile(data, 'nb_user', 0.50)
        Q3_nb_user = calculate_quartile(data, 'nb_user', 0.75)
        print("Q1 pour le nombre d'utilisateurs : ", Q1_nb_user)
        print("Q2 pour le nombre d'utilisateurs : ", Q2_nb_user)
        print("Q3 pour le nombre d'utilisateurs : ", Q3_nb_user)
        # Comparison of popularity
        fig, comparison_popularity = rating_distribution(df=data, variable='nb_user', rating_var='note_moyenne',
                                                         low_threshold=2, mean_range=(2, 3), high_threshold=4)
        print("\nComparison of Rating Distribution in %:")
        plt.show()
        print("Distribution de la note par rapport à la variable popularity en %:")
        print(comparison_popularity)
        # Conclusion
        print("Il est très net ici que les recettes ayant le moins de notes sont celles les moins bien notés. "
              "Cela veut dire qu'elles sont moins populaires et/ou moins visibles. Au contraire celles avec le plus de notes sont les mieux notées.")
        print("Ou ça peut vouloir dire que les utilisateurs ne notent pas les mauvaises recettes. La mauvaise note appelle la mauvaise note.")
        print("A CREUSER :")
        print("- qui sont les users qui ont mal noté ces recettes : ont-ils beaucoup noté ? Mettent-ils que des mauvaises notes ? "
              "Pour vérifier si cette information est significative.")
        print("- faire un heatmap : nb_users/note_moyenne")
    else:
        print("Il ne sert à rien de passer sur le data set 2 car pour cette partie car nous traçons nb_users en fonction de la note moyenne. "
              "Merci donc de revenir sur le data set 1 pour cette analyse.")


    if df1 == 1:
        bad_ratings, good_ratings = separate_bad_good_ratings(data, 4, 'note_moyenne')

        print("Analysons les tags et descriptions pour essayer de trouver des thèmes communs entre les recettes mal notées. "
              "On les comparera aux recettes bien notées. Pour cela nous utiliserons les dataframes bad_ratings et good_ratings. "
              "La première étape est de réaliser un pre-processing de ces variables (enlever les mots inutiles, tokeniser).")
        # Preprocessing des tags et descriptions
        bad_ratings.loc[:, 'tags_clean'] = bad_ratings.loc[:, 'tags'].fillna('').apply(preprocess_text)
        bad_ratings.loc[:, 'description_clean'] = bad_ratings.loc[:, 'description'].fillna('').apply(preprocess_text)
        good_ratings.loc[:, 'tags_clean'] = good_ratings.loc[:, 'tags'].fillna('').apply(preprocess_text)
        good_ratings.loc[:, 'description_clean'] = good_ratings.loc[:, 'description'].fillna('').apply(preprocess_text)
        # Mots les plus courants dans les tags des recettes mal notées
        most_common_bad_tags_clean = get_most_common_words(bad_ratings['tags_clean'])
        print("Les tags les plus courants dans les recettes mal notées :")
        bad_tag_words_set = extractWordFromTUpple(most_common_bad_tags_clean)
        print(bad_tag_words_set)
        # Mots les plus courants dans la descriptions des recettes mal notées
        most_common_bad_desciption_clean = get_most_common_words(bad_ratings['description_clean'])
        print("\nLes mots les plus courants dans les descriptions des recettes mal notées :")
        bad_desc_words_set = extractWordFromTUpple(most_common_bad_desciption_clean)
        print(bad_desc_words_set)
        # Mots les plus courants dans les tags des recettes bien notées
        most_common_good_tags_clean = get_most_common_words(good_ratings['tags_clean'])
        print("Les tags les plus courants dans les recettes bien notées :")
        good_tag_words_set = extractWordFromTUpple(most_common_good_tags_clean)
        print(good_tag_words_set)
        # Mots les plus courants dans descriptions des recettes bien notées
        most_common_good_desciption_clean = get_most_common_words(good_ratings['description_clean'])
        print("\nLes mots les plus courants dans les descriptions des recettes bien notées :")
        good_desc_words_set = extractWordFromTUpple(most_common_good_desciption_clean)
        print(good_desc_words_set)
        # Mots uniques dans les tags et descriptions des recettes mal notées :
        print("Mots uniques dans les tags des recettes mal notées :", uniqueTags(bad_tag_words_set, good_tag_words_set))
        print("Mots uniques dans les descriptions des recettes mal notées :", uniqueTags(bad_desc_words_set, good_desc_words_set))
    else:
        bad_ratings, good_ratings = separate_bad_good_ratings(data, 4, 'rating')  # la fonction marche mais en local uniquement, trop lourde en RAM pour le serveur
        print("Sur le dataset 2, les calculs prennent bcp trop de temps et d'espace RAM ce qui provoquait des crashs.")
        print("Nous avons donc fait tourner en local les calculs et les résultats sont les suivants :")
        print("Mots uniques dans les tags des recettes mal notées : {'rice', 'fish'}")
        print("Mots uniques dans les tags des recettes mal notées : {'low', 'healthy'}")
    # Conclusion
    print("Il vaut mieux éviter d'écrire une recette avec les mots et les descriptions ci-dessus.")
    print("Notons que les résultats sont différents entre les deux datasets.")

    # Page 7

    if df1 == 1:
        print("La fonction permettant de calculer le temps moyen par étape pour chaque recette ne peut fonctionner que sur le dataset 2.")
        print("Merci donc de changer de dataset.")
    else:
        fig = time_per_step(data, 'minutes', 'n_steps')
        plt.show()
        print("Plus le rapport temps par étape est élevé plus la proportion de recettes mal notée augmente.")
        # Régression linéaire
        data['minute_step'] = data['minutes'] / data['n_steps']
        X = data['minute_step']
        y = data['rating']
        model = OLS_regression(X, y)
        print(model.summary())
        print("Le modèle linéaire n'est cependant pas applicable ici. R2 est proche de 0.")

    # Page 8

    if df1 == 1:
        print("La fonction permettant de séparer les profils utilisateurs ne peut fonctionner que sur le dataset 2.")
        print("Merci donc de changer de dataset.")
    else:
        user_profiles = create_dfuser_profiles(data)
        print(user_profiles.head())
        # distribution du nombre de raters également contributeurs
        fig = rating_isContributor(user_profiles, 'is_contributor')
        plt.show()
        # Moyenne du nombre de recettes notées pour contributeurs et non-contributeurs
        contributor_stats = user_profiles.groupby('is_contributor')['num_recipes_rated'].mean()
        print("Moyenne du nombre de recettes notées pour contributeurs et non-contributeurs :")
        print(contributor_stats)
        # Distribution des notes moyennes par groupe (contributeur vs non-contributeur)
        print("Distribution des notes moyennes par groupe (contributeur vs non-contributeur)")
        fig = plot_distributionIsContributor(user_profiles, 'is_contributor', 'mean_rating')
        plt.show()
        print("Les utilisateurs ne contribuant pas sont ceux qui notent le plus mal et qui sont les plus réguliers et homogènes dans leur notation.")
        print("Les contributeurs sont ceux qui notent le plus de recettes et ils les notent bien. Cependant ils sont beaucoup plus dispersés dans leur notation. "
              "Ceci constitue un premier biais qui tire les notes et les moyennes vers le haut.")
        user_profiles = None  # Free memory


if __name__ == "__main__":
    main()
