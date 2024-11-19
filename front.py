import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rating_recipe_correlation_analysis as rrca
import nbformat
from nbconvert import HTMLExporter


def load_data(fichier):
    try:
        data = pd.read_csv(fichier)
        return data
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()
    
def append_csv(*files):
    df_list = []
    for file in files:
        df = pd.read_csv(file)
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)
    
def display_notebook(notebook_path):
    with open(notebook_path) as f:
        notebook = nbformat.read(f, as_version=4)
    html_exporter = HTMLExporter()
    (body, resources) = html_exporter.from_notebook_node(notebook)
    st.components.v1.html(body, height=800, scrolling=True)

def main():
    st.title("Analyse des mauvaises recettes") # Titre de l'application
    st.sidebar.title("Navigation") # Titre de la sidebar
    choice = st.sidebar.radio("Allez à :", ["Introduction", "Analyse pour le client", "Notebook complet"]) # Options de la sidebar

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
        # Ajouter du texte explicatif
        st.write("...")

        st.dataframe(data2.head()) # Afficher les 5 premières lignes du tableau pré-traité
        
        df=rrca.load_data(data1, data2)
        
        rrca.check_duplicates(df)
        
        figures=rrca.plot_distributions(df, ['note_moyenne', 'note_mediane'], ['Distribution de la moyenne', 'Distribution de la médiane'])
        for fig in figures:
            st.pyplot(fig)


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
    
    data1 = load_data(fichierPréTraité1) # Charger les données pré-traitées
    data2 = append_csv(fichierrecipe_cleaned_part1, fichierrecipe_cleaned_part2, fichierrecipe_cleaned_part3, fichierrecipe_cleaned_part4, fichierrecipe_cleaned_part5)
    
    main()