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

def display_fig(fig):
        st.pyplot(fig)
        plt.close()

def main():
    st.title("Analyse des mauvaises recettes") # Titre de l'application
    st.sidebar.title("Navigation") # Titre de la sidebar
    choice = st.sidebar.radio("Allez à :", ["Analyse pour le client", "Introduction", "Analyse pour le client", "Notebook complet"]) # Options de la sidebar

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
    
#############################################################################################################################################
################################## Recupération du fichier rating_recipe_correlation_analysis.py #########################################
#############################################################################################################################################   
    elif choice == "Analyse pour le client":
        st.subheader(f"Tableau pré-traité 2 : ")
        
        st.write("Affichage des 5 premières lignes de notre JDD : ")
        df=rrca.merged_data(data1, data2) # Fusionner les deux tableaux
        nb_doublon=rrca.check_duplicates(df) # Vérifier les doublons
        st.write(f"Nombre de doublons : {nb_doublon}")
        rrca.drop_columns(df, ['recipe_id', 'nutrition', 'steps']) # Supprimer les colonnes en double
        df.columns = ['name', 'recipe_id', 'minutes', 'contributor_id', 'submitted', 'tags', 'n_steps', 
                      'description', 'ingredients', 'n_ingredients', 'calories', 'total_fat', 'sugar', 
                      'sodium','protein', 'saturated_fat', 'carbohydrates', 'year', 'month', 'day', 
                      'day_of_week', 'nb_user', 'note_moyenne','note_mediane', 'note_q1', 'note_q2', 
                      'note_q3', 'note_q4', 'note_max', 'note_min', 'nb_note_lt_5', 'nb_note_eq_5']
        st.dataframe(df.head()) # Afficher les 5 premières lignes du tableau pré-traité

        # Distibution de la moyenne des notes
        st.write("Distrubution de la moyenne des notes : ")
        display_fig(rrca.plot_distribution(df, 'note_moyenne', 'Distribution de la moyenne'))

        # Distibution de la médiane des notes
        st.write("Distrubution de la médiane des notes : ")
        display_fig(rrca.plot_distribution(df, 'note_mediane', 'Distribution de la médiane'))
        
        st.subheader("Qu'est-ce qui caractérise une mauvaise recette ? : ")
        st.write("La première partie de l'analyse portera sur l'analyse des contributions qui ont eu une moyenne de moins de 4/5 ou égale à 4 :")
        st.write("Quels sont les critères d'une mauvaise recette/contribution ?")
        st.write("Quelles sont les caractéristiques des recettes les moins populaires ?")
        st.write("Qu'est-ce qui fait qu'une recette est mal notée?")

        # Matrice de corrélation
        display_fig(rrca.plot_correlation_matrix(df, ['note_moyenne', 'minutes', 'n_steps', 'n_ingredients', 'calories', 'total_fat', 
                         'sugar', 'sodium','protein', 'saturated_fat', 'carbohydrates', 'nb_user'], 
                         "Matrice de corrélation entre la moyenne et la médiane des notes"))
        st.write("Pas de corrélation entre les notes et les variables sélectionnées dans la correlation matrix.")
        st.write("Les outliers peuvent grandement affecter les corrélations. Nous avons vu qu'ils étaient nombreux")
        st.write("dans la première partie de l'analyse du dataset recipe. Nous allons les supprimer pour la suite de l'analyse.")
        
        # Boxplot df
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        # for colonne in numerical_cols:
        #     display_fig(rrca.boxplot_numerical_cols(df, colonne))

        # Suppression des outliers
        st.write("Suppression des outliers : ")
        infoOultiers=rrca.calculate_outliers(df, numerical_cols)
        st.write(infoOultiers)
        col_to_clean = ['minutes', 'n_steps', 'n_ingredients', 'calories', 'total_fat', 'sugar', 'sodium', 'protein', 'saturated_fat', 'carbohydrates']
        df_cleaned=rrca.remove_outliers(df, col_to_clean)
        st.write(f"Taille initiale du DataFrame : {df.shape}")
        st.write(f"Taille après suppression des outliers : {df_cleaned.shape}")

        # Matrice de corrélation df_cleaned
        st.write("Regardons à nouveau la matrice de corrélation et les boxplots :")
        display_fig(rrca.plot_correlation_matrix(df_cleaned, ['note_moyenne', 'minutes', 'n_steps', 'n_ingredients', 'calories', 'total_fat', 
                         'sugar', 'sodium','protein', 'saturated_fat', 'carbohydrates', 'nb_user'], 
                         "Matrice de corrélation entre la moyenne et la médiane des notes"))
        

        # Boxplot df_cleaned
        # for colonne in numerical_cols:
        #     display_fig(rrca.boxplot_numerical_cols(df_cleaned, colonne))
        #     
        st.write("Toujours pas de corrélations avec notre variable note_moyenne. Il se peut que le passage à la moyenne altère les corrélations.",
                 "Continuous l'analyse en comparant des metrics pour les good et bad ratings, nous reviendrons à ce problème de moyenne dans un deuxième temps.")
        st.write("Regardons à quelle note correspond le 1e quartile. Nous nous concentrerons sur les 25% moins bonnes recettes pour notre analyse.")

        # Calcul des quartiles
        mean_third_quartile=rrca.calculate_third_quartiles(df_cleaned, 'note_moyenne')
        st.write("3e Quartile pour la moyenne:", mean_third_quartile)
        mean_third_quartile=rrca.calculate_third_quartiles(df_cleaned, 'note_mediane')
        st.write("3e Quartile pour la médiane:", mean_third_quartile)

        # Nombre de mauvaises notes
        bad_ratings, good_ratings =rrca.separate_bad_good_ratings(df_cleaned, 4)
        st.write(f"Nombre de recettes avec une moyenne inférieure à 4 : {df_cleaned[df_cleaned['note_moyenne'] <= 4.0].shape[0]}")
        st.write(f"Nombre de recettes avec une médiane inférieure à 4 : {df_cleaned[df_cleaned['note_mediane'] <= 4.0].shape[0]}")
        st.write("Nous nous concentrerons sur la moyenne qui nous permet d'augmenter l'échantillon de bad ratings. Compte tenu de la distribution de la moyenne, on peut considérer les 4 (et moins) comme des mauvaises notes.")
        
        # Filtrer les recettes avec une note inférieure ou égale à 4 :
        st.write("Afin de comparer les recettes mal notées des bien notées, nous devons filtrer le dataframe sur les mauvaises notes (première ligne) et les bonnes notes (deuxième ligne). ")
        display_fig(rrca.plot_bad_ratings_distributions(bad_ratings, good_ratings))
        st.write("Pas de grosses variations à observer... Regardons maintenant si la saisonnalité / la période où la recette est postée a un impact :)")

        # Saisonalité
        st.write("Saisonalié des recettes mal notées (en haut) et bien notées (en bas) : ")
        display_fig(rrca.saisonnalite(bad_ratings))
        display_fig(rrca.saisonnalite(good_ratings))
        st.write("Nous n'observons pas d'impact de la saisonnalité du post entre bad et good ratings.")

        #Comparaison du temps, du nombre d'étapes et du nombre d'ingrédients entre les recettes bien et mal notées
        st.write("Analyser l'impact du temps de préparation and la complexité sur les notes :")
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

    fichierPréTraité1 = "Pretraitement/recipe_mark.csv"
    fichierrecipe_cleaned_part1 = "Pretraitement/recipe_cleaned_part_1.csv"
    fichierrecipe_cleaned_part2 = "Pretraitement/recipe_cleaned_part_2.csv"
    fichierrecipe_cleaned_part3 = "Pretraitement/recipe_cleaned_part_3.csv"        
    fichierrecipe_cleaned_part4 = "Pretraitement/recipe_cleaned_part_4.csv"
    fichierrecipe_cleaned_part5 = "Pretraitement/recipe_cleaned_part_5.csv"
    
    data1 = rrca.load_data(fichierPréTraité1) # Charger les données pré-traitées
    data2 = rrca.append_csv(fichierrecipe_cleaned_part1, fichierrecipe_cleaned_part2, fichierrecipe_cleaned_part3, fichierrecipe_cleaned_part4, fichierrecipe_cleaned_part5)
    
    main()