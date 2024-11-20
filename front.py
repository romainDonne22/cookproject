import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rating_recipe_correlation_analysis as rrca
import nbformat
from nbconvert import HTMLExporter

  

def display_notebook(notebook_path):
    with open(notebook_path) as f:
        notebook = nbformat.read(f, as_version=4)
    html_exporter = HTMLExporter()
    (body, resources) = html_exporter.from_notebook_node(notebook)
    st.components.v1.html(body, height=800, scrolling=True)

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
        fig=rrca.plot_distribution(df, 'note_moyenne', 'Distribution de la moyenne')
        st.pyplot(fig)
        plt.close()

        # Distibution de la médiane des notes
        st.write("Distrubution de la médiane des notes : ")
        fig=rrca.plot_distribution(df, 'note_mediane', 'Distribution de la médiane')
        st.pyplot(fig)
        plt.close()
        
        st.subheader("Qu'est-ce qui caractérise une mauvaise recette ? : ")
        st.write("La première partie de l'analyse portera sur l'analyse des contributions qui ont eu une moyenne de moins de 4/5 ou égale à 4 :")
        st.write("Quels sont les critères d'une mauvaise recette/contribution ?")
        st.write("Quelles sont les caractéristiques des recettes les moins populaires ?")
        st.write("Qu'est-ce qui fait qu'une recette est mal notée?")

        # Matrice de corrélation
        fig=rrca.plot_correlation_matrix(df, ['note_moyenne', 'minutes', 'n_steps', 'n_ingredients', 'calories', 'total_fat', 
                         'sugar', 'sodium','protein', 'saturated_fat', 'carbohydrates', 'nb_user'], 
                         "Matrice de corrélation entre la moyenne et la médiane des notes") 
        st.pyplot(fig)
        plt.close()
        st.write("Pas de corrélation entre les notes et les variables sélectionnées dans la correlation matrix.")
        st.write("Les outliers peuvent grandement affecter les corrélations. Nous avons vu qu'ils étaient nombreux")
        st.write("dans la première partie de l'analyse du dataset recipe. Nous allons les supprimer pour la suite de l'analyse.")
        
        # Boxplot df
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for colonne in numerical_cols:
            fig=rrca.boxplot_numerical_cols(df, colonne)
            st.pyplot(fig)
            plt.close()

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
        fig=rrca.plot_correlation_matrix(df_cleaned, ['note_moyenne', 'minutes', 'n_steps', 'n_ingredients', 'calories', 'total_fat', 
                         'sugar', 'sodium','protein', 'saturated_fat', 'carbohydrates', 'nb_user'], 
                         "Matrice de corrélation entre la moyenne et la médiane des notes") 
        st.pyplot(fig)
        plt.close()

        # Boxplot df_cleaned
        for colonne in numerical_cols:
            fig=rrca.boxplot_numerical_cols(df_cleaned, colonne)
            st.pyplot(fig)
            plt.close()
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
        fig=rrca.plot_bad_ratings_distributions(bad_ratings, good_ratings)
        st.pyplot(fig)
        plt.close()
        st.write("Pas de grosses variations à observer... Regardons maintenant si la saisonnalité / la période où la recette est postée a un impact :)")

        # Saisonalité
        st.write("Saisonalié des recettes mal notées (en haut) et bien notées (en bas) : ")
        fig = rrca.saisonnalite(bad_ratings)
        st.pyplot(fig)
        plt.close()
        fig = rrca.saisonnalite(good_ratings)
        st.pyplot(fig)
        plt.close()

        #à faire 28


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