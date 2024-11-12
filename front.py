import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

fichierPréTraité1 = "recette_statistiques.csv"

def load_data(fichier):
    try:
        data = pd.read_csv(fichier)
        return data
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()

def main():
    st.title("Analyse des mauvaises recettes")

    menu = ["Introduction","Analyse"] # Menu de l'application sur le coté gauche
    choice = st.sidebar.selectbox("Menu", menu)

    # affichage de la page Analyse des mauvaises recettes
    if choice == "Analyse":
        st.subheader("Tableau pré-traité")
        data = load_data(fichierPréTraité1) # Charger les données pré-traitées
        if not data.empty:
            st.dataframe(data)

            # Ajouter une barre de sélection glissante pour filtrer les notes moyennes
            min_note, max_note = st.slider(
                "Sélectionnez la plage de notes moyennes",
                min_value=float(data['note_moyenne'].min()),
                max_value=float(data['note_moyenne'].max()),
                value=(float(data['note_moyenne'].min()), float(data['note_moyenne'].max()))
            )

            # Filtrer les données en fonction de la plage de notes moyennes sélectionnée
            filtered_data = data[(data['note_moyenne'] >= min_note) & (data['note_moyenne'] <= max_note)]

            # Diagramme en barres pour la distribution des notes moyennes filtrées
            plt.figure(figsize=(10, 6))
            sns.histplot(filtered_data['note_moyenne'], bins=20, kde=False)
            plt.xlabel('Note Moyenne')
            plt.ylabel('Nombre de Recettes')
            plt.title('Distribution des Notes Moyennes')
            st.pyplot(plt)
        else:
            st.write("No data available")

    elif choice == "Introduction":
        st.subheader("Introduction")
        st.write("Bienvenu sur notre application qui permet d'analyser les mauvaises recettes.")
        st.subheader("Auteurs")
        st.write("- Aude De Fornel")
        st.write("- Camille Ishac")
        st.write("- Romain Donné")

if __name__ == "__main__":
    main()