import streamlit as st
import pandas as pd
import time

# Exemple de fonction qui charge des données à partir d'un fichier CSV
# Utilisation de st.cache_data pour mettre en cache les données
@st.cache_data
def load_data():
    # Simuler un chargement de données lent (par exemple depuis un fichier ou une base de données)
    time.sleep(2)  # Simuler un délai de 2 secondes pour le chargement des données
    data = pd.DataFrame({
        "Nom": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "Âge": [25, 30, 35, 40, 45],
        "Ville": ["Paris", "Lyon", "Marseille", "Lille", "Bordeaux"]
    })
    return data

# Chargement des données à l'aide de la fonction mise en cache
st.title("Exemple d'application Streamlit avec st.cache_data")

# Bouton pour charger les données
if st.button("Charger les données"):
    st.write("Chargement des données...")
    data = load_data()  # Appel à la fonction de chargement des données
    st.dataframe(data)  # Afficher les données dans un tableau

# Explication de l'utilisation du cache
st.write(
    "Les données sont mises en cache grâce à `st.cache_data`, "
    "ce qui signifie que si vous rechargez les données ou réagissez à d'autres interactions, "
    "les données ne seront pas rechargées depuis zéro."
)