import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

fichierPréTraité1 = "Pretraitement/recette_statistiques.csv"
fichierPréTraité2 = "Pretraitement/recipe_cleaned.csv"


def load_data(fichier):
    try:
        data = pd.read_csv(fichier)
        return data
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()

def main():
    st.title("Analyse des mauvaises recettes") # Titre de l'application

    menu = ["Introduction","Analyse"] # Menu de l'application sur le coté gauche
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
################################## Recupération du fichier rating_recipe_correlation_analysis.ipynb #########################################
#############################################################################################################################################   
        st.subheader(f"Tableau pré-traité : {fichierPréTraité2}")
        # Ajouter du texte explicatif
        st.write("...")
        data2 = load_data(fichierPréTraité2) # Charger les données pré-traitées
        merged_data = pd.merge(data2, data1, left_on="id", right_on="recipe_id", how="left")
        # Dropons la colonne id en double et la colonne nutrition déjà traitée
        merged_data.drop(['recipe_id','nutrition','steps'], axis=1, inplace=True)
        merged_data.columns = ['name', 'recipe_id', 'minutes', 'contributor_id', 'submitted', 'tags',
       'n_steps', 'description', 'ingredients',
       'n_ingredients', 'calories', 'total_fat', 'sugar', 'sodium',
       'protein', 'saturated_fat', 'carbohydrates', 'year',
       'month', 'day', 'day_of_week', 'nb_user', 'note_moyenne',
       'note_mediane', 'note_q1', 'note_q2', 'note_q3', 'note_q4', 'note_max',
       'note_min', 'nb_note_lt_5', 'nb_note_eq_5']
        merged_data['recipe_id'] = merged_data['recipe_id'].astype('object')
        merged_data['contributor_id'] = merged_data['recipe_id'].astype('object')
        merged_data['year'] = merged_data['year'].astype('object')
        merged_data['month'] = merged_data['month'].astype('object')
        merged_data['day'] = merged_data['day'].astype('object')
        plt.figure(figsize=(12, 5))

        # Distribution de la moyenne
        plt.subplot(1, 2, 1)
        plt.hist(merged_data['note_moyenne'], bins=20)
        plt.title('Distribution de la moyenne')
        plt.xlabel('Moyenne')
        plt.ylabel('Fréquence')

        # Distribution de la médiane
        plt.subplot(1, 2, 2)
        plt.hist(merged_data['note_mediane'], bins=20)
        plt.title('Distribution de la médiane')
        plt.xlabel('Médiane')
        plt.ylabel('Fréquence')

        plt.tight_layout()
        st.pyplot(plt)

        # Commençons par regarder les corrélations grâce à une matrice de corrélation
        plt.figure(figsize=(12, 8))
        correlation = merged_data[['note_moyenne', 'minutes', 'n_steps', 'n_ingredients', 'calories', 'total_fat', 
                                'sugar', 'sodium','protein', 'saturated_fat', 'carbohydrates', 'nb_user']].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm')
        plt.title('Matrice de correlation des variables avec la moyenne')
        st.pyplot(plt)



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

if __name__ == "__main__":
    main()