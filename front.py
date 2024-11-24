import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rating_recipe_correlation_analysis as rrca
import nbformat
import numpy as np
from nbconvert import HTMLExporter

  

def display_notebook(notebook_path):
    with open(notebook_path) as f:
        notebook = nbformat.read(f, as_version=4)
    html_exporter = HTMLExporter()
    (body, resources) = html_exporter.from_notebook_node(notebook)
    st.components.v1.html(body, height=800, scrolling=True)

# Fonction pour alterner entre les DataFrames
def toggle_dataframe():
        st.session_state.df_index = 1 - st.session_state.df_index # On alterne l'index du DataFrame affiché

# Fonction pour afficher les figures et les fermer après affichage afin de libérer la mémoire
def display_fig(fig):
        st.pyplot(fig)
        plt.close()

# Charger le premier JDD (notes moyennées) une seule fois en cache sur le serveur Streamlit Hub
@st.cache_data 
def init_data_part1():
    data1 = rrca.load_csv("Pretraitement/recipe_mark.csv")
    data2 = rrca.append_csv(
                "Pretraitement/recipe_cleaned_part_1.csv",
                "Pretraitement/recipe_cleaned_part_2.csv",
                "Pretraitement/recipe_cleaned_part_3.csv",
                "Pretraitement/recipe_cleaned_part_4.csv",
                "Pretraitement/recipe_cleaned_part_5.csv")
    df = rrca.merged_data(data1, data2) 
    rrca.drop_columns(df, ['recipe_id', 'nutrition', 'steps']) # Supprimer les colonnes en double
    df.columns = ['name', 'recipe_id', 'minutes', 'contributor_id', 'submitted', 'tags', 'n_steps', 
                    'description', 'ingredients', 'n_ingredients', 'calories', 'total_fat', 'sugar', 
                    'sodium','protein', 'saturated_fat', 'carbohydrates', 'year', 'month', 'day', 
                    'day_of_week', 'nb_user', 'note_moyenne','note_mediane', 'note_q1', 'note_q2', 
                    'note_q3', 'note_q4', 'note_max', 'note_min', 'nb_note_lt_5', 'nb_note_eq_5'] # Renommer les colonnes
    col_to_clean = ['minutes', 'n_steps', 'n_ingredients', 'calories', 'total_fat', 'sugar', 'sodium', 'protein', 'saturated_fat', 'carbohydrates']
    df_cleaned=rrca.remove_outliers(df, col_to_clean)
    data1 = None # Libérer la mémoire
    data2 = None # Libérer la mémoire
    df = None # Libérer la mémoire
    return df_cleaned

# Charger le deucième JDD (toutes les notes) une seule fois en cache sur le serveur Streamlit Hub
@st.cache_data
def init_data_part2():
    data2 = rrca.append_csv(
                    "Pretraitement/recipe_cleaned_part_1.csv",
                    "Pretraitement/recipe_cleaned_part_2.csv",
                    "Pretraitement/recipe_cleaned_part_3.csv",
                    "Pretraitement/recipe_cleaned_part_4.csv",
                    "Pretraitement/recipe_cleaned_part_5.csv")
    data3 = rrca.append_csv(
                    "Pretraitement/RAW_interactions_part_1.csv",
                    "Pretraitement/RAW_interactions_part_2.csv",
                    "Pretraitement/RAW_interactions_part_3.csv",
                    "Pretraitement/RAW_interactions_part_4.csv",
                    "Pretraitement/RAW_interactions_part_5.csv")
    user_analysis = pd.merge(data3, data2, left_on="recipe_id", right_on="id", how="left")
    data2 = None # Libérer la mémoire
    data3 = None # Libérer la mémoire
    user_analysis = user_analysis.dropna(subset=['name']) # 34 notes ne correspondent à aucune recette. Ce sont les outliers qu'on a sorti du dataset recipe lors de la première analyse. Nous allons les drop.
    user_analysis['review'] = user_analysis['review'].fillna("missing")
    # Nous ne gardons que les colonnes utiles à l'analyse et non répétitive
    user_analysis.drop(['name', 'id','nutrition','steps', 'saturated fat (%)'], axis=1, inplace=True)
    id_columns = ['recipe_id', 'user_id', 'contributor_id','year', 'month', 'day']
    for col in id_columns:
        user_analysis[col] = user_analysis[col].astype('object')
    # Renaming des colonnes :
    user_analysis.columns = ['user_id', 'recipe_id', 'date', 'rating', 'review', 'minutes',
            'contributor_id', 'submitted', 'tags', 'n_steps', 'description',
            'ingredients', 'n_ingredients', 'calories', 'total_fat',
            'sugar', 'sodium', 'protein', 'carbohydrates', 'year',
            'month', 'day', 'day_of_week']
    
    # Créer la variable binaire cible 'binary_rating' en fonction de la note
    # Mauvaise note (<=4) sera codée par 0, et bonne note (>4) par 1
    user_analysis['binary_rating'] = user_analysis['rating'].apply(lambda x: 0 if x <= 4 else 1)
    numerical_col = user_analysis.select_dtypes(include=['int64', 'float64']).columns

    # Calculer les pourcentages d'outliers pour chaque colonne :
    outlier_info = {}
    for column in numerical_col:  
        Q1 = user_analysis[column].quantile(0.15)  
        Q3 = user_analysis[column].quantile(0.85)  
        IQR = Q3 - Q1  # Étendue interquartile
        # Nous nous concentrerons sur la borne supérieure qui comprend tous les outliers
        upper_bound = Q3 + 1.5 * IQR
        # Identifier les outliers
        outliers = user_analysis[(user_analysis[column] > upper_bound)]
        # Calculer le pourcentage d'outliers
        outlier_percentage = (len(outliers) / len(user_analysis)) * 100
        # Ajouter les informations dans un dictionnaire
        outlier_info[column] = {'Upper Bound': upper_bound,'Outlier Count': len(outliers),'Outlier Percentage (%)': outlier_percentage}
    # Définir les colonnes sur lesquelles nous voulons appliquer la suppression des valeurs aberrantes
    col_to_clean = ['minutes', 'n_steps', 'n_ingredients', 'calories', 'total_fat', 'sugar','sodium', 'protein', 'carbohydrates']
    # Créer une copie du DataFrame initial pour travailler dessus
    cleaned_user_analysis = user_analysis.copy()
    # Appliquer le filtrage des outliers
    for col in col_to_clean:
        Q1 = user_analysis[col].quantile(0.15)
        Q3 = user_analysis[col].quantile(0.85)
        IQR = Q3 - Q1  # Étendue interquartile
        # Calculer la borne supérieure
        upper_bound = Q3 + 1.5 * IQR
        # Filtrer les lignes cumulativement
        cleaned_user_analysis = cleaned_user_analysis[cleaned_user_analysis[col] <= upper_bound]
    #==> pas assez de mémoire, la solution, n'utiliser que cleaned_user_analysis et faisant le prétraitment avant.
    user_analysis = None # Libérer la mémoire

    return cleaned_user_analysis
    
def main():
    st.title("Analyse des mauvaises recettes") # Titre de l'application
    df_cleaned = init_data_part1() # Charger les données du premier JDD
    cleaned_user_analysis = init_data_part2() # Charger les données du deuxième JDD
    st.sidebar.title("Navigation") # Titre de la sidebar
    choice = st.sidebar.radio("Allez à :", ["Introduction", "Caractéristiques des recettes mal notées", 
        "Influence du temps de préparation et de la complexité", "Influence du contenu nutritionnel", 
        "Influence de popularité et de la visibilité", "Influence des tags et des descriptions"]) # Options de la sidebar
    if 'df_index' not in st.session_state:
        st.session_state.df_index = 0  # Initialisation pour afficher df1 au départ
    st.sidebar.button('Changer de DataFrame', on_click=toggle_dataframe) # Affichage du bouton pour alterner
    if st.session_state.df_index == 0: # Affichage du DataFrame sélectionné en fonction de l'état
        st.sidebar.write("Le DataFrame 1 est sélectionné, c'est à dire celui avec les notes moyennes par recettes")
        st.sidebar.write(st.session_state.df_index)
        data = df_cleaned
    else:
        st.sidebar.write("Le DataFrame 2 est sélectionné, c'est à dire celui avec toutes les notes par recettes")
        st.sidebar.write(st.session_state.df_index)
        data = cleaned_user_analysis
    


#############################################################################################################################################
################################## Affichage de la page introduction ########################################################################
############################################################################################################################################# 
    if choice == "Introduction":
        st.subheader("Introduction")
        st.write("Bienvenu sur notre application qui permet d'analyser les mauvaises recettes.")
        st.subheader("Auteurs")
        st.write("- Aude De Fornel")
        st.write("- Camille Ishac")
        st.write("- Romain Donné")
        st.write("Lien du GitHub : https://github.com/romainDonne22/cookproject")
    
#############################################################################################################################################
################################## Recupération du fichier rating_recipe_correlation_analysis.py #########################################
#############################################################################################################################################   
    elif choice == "Caractéristiques des recettes mal notées":
        st.subheader("Qu'est-ce qui caractérise une mauvaise recette ?")
        st.write("Affichons des 5 premières lignes de notre JDD : ")
        st.dataframe(data.head()) # Afficher les 5 premières lignes du tableau pré-traité
        nb_doublon=rrca.check_duplicates(data) # Vérifier les doublons
        st.write(f"Nombre de doublons : {nb_doublon}")
        
        if st.session_state.df_index == 0 :
            # Distibution de la moyenne des notes
            st.write("Distrubution de la moyenne des notes : ")
            display_fig(rrca.plot_distribution(data, 'note_moyenne', 'Distribution de la moyenne'))
            # Distibution de la médiane des notes
            st.write("Distrubution de la médiane des notes : ")
            display_fig(rrca.plot_distribution(data, 'note_mediane', 'Distribution de la médiane'))
            
        else:
            display_fig(rrca.plot_distribution(data, 'rating', 'Distribution de la moyenne'))

        
        # st.subheader("Qu'est-ce qui caractérise une mauvaise recette ? : ")
        # st.write("La première partie de l'analyse portera sur l'analyse des contributions qui ont eu une moyenne de moins de 4/5 ou égale à 4 :")
        # st.write("Quels sont les critères d'une mauvaise recette/contribution ?")
        # st.write("Quelles sont les caractéristiques des recettes les moins populaires ?")
        # st.write("Qu'est-ce qui fait qu'une recette est mal notée?")

        # # Matrice de corrélation
        # display_fig(rrca.plot_correlation_matrix(df, ['note_moyenne', 'minutes', 'n_steps', 'n_ingredients', 'calories', 'total_fat', 
        #                  'sugar', 'sodium','protein', 'saturated_fat', 'carbohydrates', 'nb_user'], 
        #                  "Matrice de corrélation entre la moyenne et la médiane des notes"))
        # st.write("Pas de corrélation entre les notes et les variables sélectionnées dans la correlation matrix.")
        # st.write("Les outliers peuvent grandement affecter les corrélations. Nous avons vu qu'ils étaient nombreux")
        # st.write("dans la première partie de l'analyse du dataset recipe. Nous allons les supprimer pour la suite de l'analyse.")
        
        # # Boxplot df
        # numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
        # for colonne in numerical_cols:
        #     display_fig(rrca.boxplot_numerical_cols(df, colonne))

        # # Suppression des outliers
        # st.write("Suppression des outliers : ")
        # infoOultiers=rrca.calculate_outliers(df, numerical_cols)
        # st.write(infoOultiers)
        # st.write(f"Taille initiale du DataFrame : {df.shape}")
        st.write(f"Taille après suppression des outliers : {data.shape}")

        # Matrice de corrélation df_cleaned
        st.write("Regardons à nouveau la matrice de corrélation et les boxplots :")
        display_fig(rrca.plot_correlation_matrix(data, ['note_moyenne', 'minutes', 'n_steps', 'n_ingredients', 'calories', 'total_fat', 
                         'sugar', 'sodium','protein', 'saturated_fat', 'carbohydrates', 'nb_user'], 
                         "Matrice de corrélation entre la moyenne et la médiane des notes"))
        
        # Boxplot df_cleaned
        for colonne in numerical_cols:
            display_fig(rrca.boxplot_numerical_cols(df_cleaned, colonne))
             
        st.write("Toujours pas de corrélations avec notre variable note_moyenne. Il se peut que le passage à la moyenne altère les corrélations.",
                 "Continuous l'analyse en comparant des metrics pour les good et bad ratings, nous reviendrons à ce problème de moyenne dans un deuxième temps.")
        st.write("Regardons à quelle note correspond le 1e quartile. Nous nous concentrerons sur les 25% moins bonnes recettes pour notre analyse.")

        # Calcul des quartiles
        mean_quartile=rrca.calculate_quartile(df_cleaned, 'note_moyenne', 0.25)
        st.write("3e Quartile pour la moyenne:", mean_quartile)
        mean_quartile=rrca.calculate_quartile(df_cleaned, 'note_mediane', 0.25)
        st.write("3e Quartile pour la médiane:", mean_quartile)

        # Nombre de mauvaises notes
        st.write(f"Nombre de recettes avec une moyenne inférieure à 4 : {df_cleaned[df_cleaned['note_moyenne'] <= 4.0].shape[0]}")
        st.write(f"Nombre de recettes avec une médiane inférieure à 4 : {df_cleaned[df_cleaned['note_mediane'] <= 4.0].shape[0]}")
        st.write("Nous nous concentrerons sur la moyenne qui nous permet d'augmenter l'échantillon de bad ratings. Compte tenu de la distribution de la moyenne, on peut considérer les 4 (et moins) comme des mauvaises notes.")
        
        # Filtrer les recettes avec une note inférieure ou égale à 4 :
        bad_ratings, good_ratings = rrca.separate_bad_good_ratings(df_cleaned, 4)
        st.write("Afin de comparer les recettes mal notées des bien notées, nous devons filtrer le dataframe sur les mauvaises notes (première ligne) et les bonnes notes (deuxième ligne). ")
        display_fig(rrca.plot_bad_ratings_distributions(bad_ratings, good_ratings))
        st.write("Pas de grosses variations à observer... Regardons maintenant si la saisonnalité / la période où la recette est postée a un impact :)")

        # Saisonalité
        st.write("Saisonalié des recettes mal notées (en haut) et bien notées (en bas) : ")
        display_fig(rrca.saisonnalite(bad_ratings))
        display_fig(rrca.saisonnalite(good_ratings))
        st.write("Nous n'observons pas d'impact de la saisonnalité du post entre bad et good ratings.")

    elif choice == "Influence du temps de préparation et de la complexité":
        #Comparaison du temps, du nombre d'étapes et du nombre d'ingrédients entre les recettes bien et mal notées
        st.subheader("Analyser l'impact du temps de préparation and la complexité sur les notes :")
        bad_ratings, good_ratings = rrca.separate_bad_good_ratings(df_cleaned, 4)
        data_minutes = [good_ratings['minutes'], bad_ratings['minutes']]
        display_fig(rrca.boxplot_df(data_minutes))
        data_steps = [good_ratings['n_steps'], bad_ratings['n_steps']]
        display_fig(rrca.boxplot_df(data_steps))
        data_ingred = [good_ratings['n_ingredients'], bad_ratings['n_ingredients']]
        display_fig(rrca.boxplot_df(data_ingred))
        st.write("Les recettes mal notées tendent à avoir des temps de préparation plus longs et un nombre d'étapes à suivre plus élevé. Rien à signalier sur le nombre d'ingrédients.")

        # Distribution de la note par rapport à la variable minutes / n_steps / n_ingredients en %:
        st.write("Pour aller plus loin dans l'analyse nous allons créer des bins pour chaque variable avec des seuils définis (low, medium, high) et regarder le proportion des moyennes dans chaque catégorie.")
        fig, comparison_minutes = rrca.rating_distribution(df=df_cleaned,variable='minutes',rating_var='note_moyenne',low_threshold=15,mean_range=(30, 50),high_threshold=180)
        display_fig(fig)
        st.write("Distribution de la note par rapport à la variable minutes en %:")
        st.write(comparison_minutes)
        fig, comparison_steps = rrca.rating_distribution(df=df_cleaned,variable='n_steps',rating_var='note_moyenne',low_threshold=3,mean_range=(8, 10),high_threshold=15)
        display_fig(fig)
        st.write("Distribution de la note par rapport à la variable n_steps en %:")
        st.write(comparison_steps)
        fig, comparison_ingr = rrca.rating_distribution(df=df_cleaned,variable='n_ingredients',rating_var='note_moyenne',low_threshold=3,mean_range=(8, 10),high_threshold=15)
        display_fig(fig)
        st.write("Distribution de la note par rapport à la variable n_ingredients en %:")
        st.write(comparison_ingr)
        st.write("Même analyse pour la variable nombre d'étapes : plus les recettes ont un nombre d'étapes élevé / sont complexes plus elles sont mal notées. A contrario les recettes avec moins de 3 étapes sont sensiblement mieux notées.")
        st.write("Le nombre d'ingrédients en revanche ne semble pas impacté la moyenne.")
        
        st.write("Réalisons une régression avec ces trois variables pour comprendre dans quelle mesure elles impactent la note et si cette hypothèse est statistiquement viable.")
        st.write("La matrice de corrélation en les variables 'minutes','n_steps','n_ingredients' est la suivante")
        columns_to_analyze = ['minutes','n_steps','n_ingredients']
        correlation = rrca.correlation(df_cleaned, columns_to_analyze)
        st.write(correlation)

        # Régression linéaire
        st.write("Régression linéaire entre les variables 'minutes','n_steps','n_ingredients' et la note moyenne : ")
        X = df_cleaned[['minutes', 'n_steps']]
        y = df_cleaned['note_moyenne']
        model=rrca.OLS_regression(X,y)
        st.write(model.summary())
        st.write("ANALYSE :")
        st.write("R-Squared = O.OO1 -> seulement 0.1% de la variance dans les résultats est expliquée par les variables n_steps et minutes. C'est très bas, ces variables ne semblent pas avoir de pouvoir prédictif sur les ratings, même si on a pu détecter des tendances de comportements users.")
        st.write("Prob (F-Stat) = p-value est statistiquement signifiante (car < 0.05) -> au moins un estimateur a une relation linéaire avec note_moyenne. Cependant l'effet sera minime, comme le montre le résultat R-Squared")
        st.write("Coef minute : VERY small. p-value < 0.05 donc statistiquement signifiant mais son effet est quasi négligeable sur note_moyenne. Même constat pour n_steps même si l'effet est légèrement supérieur : une augmentation de 10 étapes va baisser la moyenne d'environ 0.025...")
        st.write("Les tests Omnibus / Prob(Omnibus) et Jarque-Bera (JB) / Prob(JB) nous permettent de voir que les résidus ne suivent probablement pas une distribution gaussienne, les conditions pour une OLS ne sont donc pas remplies.")
        st.write("--> il va falloir utiliser une log transformation pour s'approcher de variables gaussiennes.")

        # 35) Régression linéaire avec log transformation
        df_cleaned['minutes_log'] = np.log1p(df_cleaned['minutes'])
        df_cleaned['n_steps_log'] = np.log1p(df_cleaned['n_steps'])
        model=rrca.OLS_regression(X,y)
        st.write(model.summary())
        st.write("ANALYSE :")
        st.write("En passant au log, on se rend compte que la variable minute a plus de poids sur la moyenne que le nombre d'étapes. Néanmoins bien que les variables minutes_log et n_steps_log soient statistiquement significatives (cf p value), leur contribution à la prédiction de la note moyenne est très faible.")
        st.write("En effet R2 est toujours extrêmement petit donc ces deux variables ont un impact minime sur la moyenne, qui ne permet pas d'expliquer les variations de la moyenne.")
        st.write("Il est probablement nécessaire d'explorer d'autres variables explicatives ou d'utiliser un modèle non linéaire pour mieux comprendre la note_moyenne.")

    elif choice == "Influence du contenu nutritionnel":
        st.subheader("Analyser le contenu nutritionnel des recettes et leur impact sur les notes")
        # comparaison calories
        fig, comparison_calories = rrca.rating_distribution(df=df_cleaned,variable='calories',rating_var='note_moyenne',low_threshold=100,mean_range=(250, 350),high_threshold=1000)
        st.write("\nComparison of Rating Distribution in %:")
        display_fig(fig)
        st.write("Distribution de la note par rapport à la variable calories en %:")
        st.write(comparison_calories)
        # comparaison total_fat
        fig, comparison_total_fat = rrca.rating_distribution(df=df_cleaned,variable='total_fat',rating_var='note_moyenne',low_threshold=8,mean_range=(15, 25),high_threshold=100)
        st.write("\nComparison of Rating Distribution in %:")
        display_fig(fig)
        st.write("Distribution de la note par rapport à la variable total_fat en %:")
        st.write(comparison_total_fat)
        # comparaison sugar
        fig, comparison_sugar = rrca.rating_distribution(df=df_cleaned,variable='sugar',rating_var='note_moyenne',low_threshold=8,mean_range=(15, 25),high_threshold=60)
        st.write("\nComparison of Rating Distribution in %:")
        display_fig(fig)
        st.write("Distribution de la note par rapport à la variable sugar en %:")
        st.write(comparison_sugar)
        # comparaison protein
        fig, comparison_protein = rrca.rating_distribution(df=df_cleaned,variable='protein',rating_var='note_moyenne',low_threshold=8,mean_range=(15, 25),high_threshold=60)
        st.write("\nComparison of Rating Distribution in %:")
        display_fig(fig)
        st.write("Distribution de la note par rapport à la variable protein en %:")
        st.write(comparison_protein)
        # conclusion
        st.write("Les variations sont trop faibles. Les contenus nutritionnels des recettes n'impactent pas la moyenne.")

    elif choice == "Influence de popularité et de la visibilité":
        #42 Analyser l'impact de la popularité des recettes sur les notes
        st.subheader("Analyser l'impact de la popularité et de la visibilité des recettes sur les notes")
        # Calculer Q1, Q3, et IQR pour le nb_users
        Q1_nb_user = rrca.calculate_quartile(df_cleaned, 'nb_user',0.25)
        Q2_nb_user = rrca.calculate_quartile(df_cleaned, 'nb_user',0.50)
        Q3_nb_user = rrca.calculate_quartile(df_cleaned, 'nb_user',0.75)
        st.write("Q1 pour le nombre d'utilisateurs : ", Q1_nb_user)
        st.write("Q2 pour le nombre d'utilisateurs : ", Q2_nb_user)
        st.write("Q3 pour le nombre d'utilisateurs : ", Q3_nb_user)
        # comparaison popularity
        fig, comparison_popularity = rrca.rating_distribution(df=df_cleaned,variable='nb_user',rating_var='note_moyenne',low_threshold=2,mean_range=(2, 3),high_threshold=4)
        st.write("\nComparison of Rating Distribution in %:")
        display_fig(fig)
        st.write("Distribution de la note par rapport à la variable popularity en %:")
        st.write(comparison_popularity)
        # conclusion
        st.write("Il est très net ici que les recettes ayant le moins de notes sont celles les moins bien notés. Cela veut dire qu'elles sont moins populaires et/ou moins visibles. Au contraire celles avec le plus de notes sont les mieux notées.")
        st.write("Ou ça peut vouloir dire que les utilisateurs ne notent pas les mauvaises recettes. La mauvaise note appelle la mauvaise note.")
        st.write("A CREUSER :")
        st.write("- qui sont les users qui ont mal noté ces recettes : ont-ils beaucoup noté ? Mettent-ils que des mauvaises notes ? Pour vérifier si cette information est significative.")
        st.write("- faire un heatmap : nb_users/note_moyenne")

    elif choice == "Influence des tags et des descriptions":
        #44 Analyser des variables categorical - tags & descriptions
        st.subheader("Analyses des variables categorical - tags & descriptions - pour comprendre grâce au verbage les critères d'une mauvaise note")
        bad_ratings, good_ratings = rrca.separate_bad_good_ratings(df_cleaned, 4)
        st.write("Analysons les tags et descriptions pour essayer de trouver des thèmes communs entre les recettes mal notées. On les comparera aux recettes bien notées. Pour cela nous utiliserons les dataframes bad_ratings et good_ratings. La première étape est de réaliser un pre-processing de ces variables (enlever les mots inutiles, tokeniser).")
        # Preprocessing des tags et descriptions
        bad_ratings.loc[:,'tags_clean'] = bad_ratings.loc[:,'tags'].fillna('').apply(rrca.preprocess_text)
        bad_ratings.loc[:,'description_clean'] = bad_ratings.loc[:,'description'].fillna('').apply(rrca.preprocess_text)
        good_ratings.loc[:,'tags_clean'] = good_ratings.loc[:,'tags'].fillna('').apply(rrca.preprocess_text)
        good_ratings.loc[:,'description_clean'] = good_ratings.loc[:,'description'].fillna('').apply(rrca.preprocess_text)
        # Mots les plus courants dans les tags des recettes mal notées
        most_common_bad_tags_clean = rrca.get_most_common_words(bad_ratings['tags_clean'])
        st.write("Les tags les plus courants dans les recettes mal notées :")
        bad_tag_words_set=rrca.extractWordFromTUpple(most_common_bad_tags_clean)
        st.write(bad_tag_words_set)
        # Mots les plus courants dans la descriptions des recettes mal notées
        most_common_bad_desciption_clean = rrca.get_most_common_words(bad_ratings['description_clean'])
        st.write("\nLes mots les plus courants dans les descriptions des recettes mal notées ::")
        bad_desc_words_set=rrca.extractWordFromTUpple(most_common_bad_desciption_clean)
        st.write(bad_desc_words_set)
        # Mots les plus courants dans les tags des recettes bien notées
        most_common_good_tags_clean = rrca.get_most_common_words(good_ratings['tags_clean'])
        st.write("Les tags les plus courants dans les recettes bien notées :")
        good_tag_words_set=rrca.extractWordFromTUpple(most_common_good_tags_clean)
        st.write(good_tag_words_set)
        # Mots les plus courants dans descriptions des recettes bien notées
        most_common_good_desciption_clean = rrca.get_most_common_words(good_ratings['description_clean'])
        st.write("\nLes mots les plus courants dans les descriptions des recettes bien notées ::")
        good_desc_words_set=rrca.extractWordFromTUpple(most_common_good_desciption_clean)
        st.write(good_desc_words_set)
        # Mots uniques dans les tags et descriptions des recettes mal notées :
        st.write("Mots uniques dans les tags des recettes mal notées :", rrca.uniqueTags(bad_tag_words_set, good_tag_words_set))
        st.write("Mots uniques dans les descriptions des recettes mal notées :", rrca.uniqueTags(bad_desc_words_set, good_desc_words_set))
        # Conclusion
        st.write("Il vaut mieux éviter d'écrire une recette avec les mots et les descriptions ci-dessus.")
        st.write("La moyenne a pu modifier les corrélations entre variables. Nous allons inverser notre dataset pour vérifier cette hypothèse : partir du dataset user et y join les informations liées aux recettes. Nous aurons ainsi une ligne par rating dans notre dataset (et non une ligne par recette comme précédemment). De cette manière les variations et préférences individuelles seront analysables. ")















#############################################################################################################################################
############################# Affichage de la page notebook #################################################################################
############################################################################################################################################# 
    elif choice == "Notebook":
        st.subheader("Notebook annalyse complète")
        notebook_path = "rating_recipe_correlation_analysis.ipynb" # Chemin du notebook
        display_notebook(notebook_path)


#############################################################################################################################################
############################### Affichage de la page preproccesse ###########################################################################
############################################################################################################################################# 
    elif choice == "Données pré-traitées":
        st.subheader("Notebook création des données pré-traitées")
        notebook_path2 = "Pretraitement/RAW_recipes to recipe_cleaned.ipynb" # Chemin du notebook
        display_notebook(notebook_path2)



if __name__ == "__main__":
    main()