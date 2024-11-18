import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rating_recipe_correlation_analysis as rrca
import nbformat
from nbconvert import HTMLExporter

fichierPréTraité1 = "Pretraitement/recipe_mark.csv"
fichierPréTraité2 = "Pretraitement/recipe_cleaned.csv"


def load_data(fichier):
    try:
        data = pd.read_csv(fichier)
        return data
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()
    
def display_notebook(notebook_path):
    with open(notebook_path) as f:
        notebook = nbformat.read(f, as_version=4)
    html_exporter = HTMLExporter()
    (body, resources) = html_exporter.from_notebook_node(notebook)
    st.components.v1.html(body, height=800, scrolling=True)

def main():
    st.title("Analyse des mauvaises recettes") # Titre de l'application

    # Premier menu de l'application sur le côté gauche
    menu = ["Introduction","Analyse", "Notebook", "RAW_interactions to recipe_mark","RAW_recipes to recipe_cleaned"] # Menu de l'application sur le coté gauche
    choice = st.sidebar.radio("Menu", menu) # Barre de sélection pour choisir la page à afficher
   
    # affichage de la page Analyse des mauvaises recettes
    if choice == "Analyse":
        st.subheader(f"Tableau pré-traité : {fichierPréTraité1}")
        # Ajouter du texte explicatif
        st.write("Le fichier pré-traité contient la note moyenne, médiane, ecart-type des recettes.")
        data1 = load_data(fichierPréTraité1) # Charger les données pré-traitées
        if not data1.empty:
            st.dataframe(data1)

            # Ajouter une barre de sélection glissante pour filtrer les notes moyennes
            min_note, max_note = st.slider(
                "Sélectionnez la plage de notes moyennes",
                min_value=float(data1['note_moyenne'].min()),
                max_value=float(data1['note_moyenne'].max()),
                value=(float(data1['note_moyenne'].min()), float(data1['note_moyenne'].max()))
            )

            # Filtrer les données en fonction de la plage de notes moyennes sélectionnée
            filtered_data = data1[(data1['note_moyenne'] >= min_note) & (data1['note_moyenne'] <= max_note)]

            # Diagramme en barres pour la distribution des notes moyennes filtrées
            plt.figure(figsize=(10, 6))
            sns.histplot(filtered_data['note_moyenne'], bins=20, kde=False)
            plt.xlabel('Note Moyenne')
            plt.ylabel('Nombre de Recettes')
            plt.title('Distribution des Notes Moyennes')
            st.pyplot(plt)
        else:
            st.write("No data available")

#############################################################################################################################################
################################## Recupération du fichier rating_recipe_correlation_analysis.py #########################################
#############################################################################################################################################   
        st.subheader(f"Tableau pré-traité : {fichierPréTraité2}")
        # Ajouter du texte explicatif
        st.write("...")
        fichierMerged=rrca.load_data(fichierPréTraité1, fichierPréTraité2)
        figures=rrca.analysisData(fichierMerged)
        for fig in figures:
                st.pyplot(fig)


#############################################################################################################################################
################################## Affichage de la page introduction ########################################################################
############################################################################################################################################# 
    elif choice == "Introduction":
        st.subheader("Introduction")
        st.write("Bienvenu sur notre application qui permet d'analyser les mauvaises recettes.")
        st.subheader("Auteurs")
        st.write("- Aude De Fornel")
        st.write("- Camille Ishac")
        st.write("- Romain Donné")

#############################################################################################################################################
############################### Affichage de la page notebook ##############################################################################
############################################################################################################################################# 
    elif choice == "Notebook":
        st.subheader("Notebook")
        notebook_path = "rating_recipe_correlation_analysis.ipynb" # Chemin du notebook
        display_notebook(notebook_path)

if __name__ == "__main__":
    main()