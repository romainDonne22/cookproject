import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

API_URL = "http://127.0.0.1:8000"

def get_recipes():
    response = requests.get(f"{API_URL}/recipes")
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Failed to fetch recipes")
        return []

def add_recipe(name, description, rating, ingredients):
    recipe = {
        "name": name,
        "description": description,
        "rating": rating,
        "ingredients": ingredients
    }
    response = requests.post(f"{API_URL}/recipes", json=recipe)
    if response.status_code == 201:
        st.success("Recipe added successfully")
    else:
        st.error("Failed to add recipe")

def load_data():
    try:
        data = pd.read_csv("recette_statistiques.csv")
        return data
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()

def main():
    st.title("Recipe Manager")

    menu = ["View Recipes", "Add Recipe"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "View Recipes":
        st.subheader("View Recipes")
        data = load_data()
        if not data.empty:
            st.dataframe(data)

            # diagramme en barres pour la distribution des notes moyennes
            plt.figure(figsize=(10, 6))
            sns.histplot(data['note_moyenne'], bins=20, kde=False)
            plt.xlabel('Note Moyenne')
            plt.ylabel('Nombre de Recettes')
            plt.title('Distribution des Notes Moyennes')
            st.pyplot(plt)
        else:
            st.write("No data available")

    elif choice == "Add Recipe":
        st.subheader("Add Recipe")
        name = st.text_input("Name")
        description = st.text_area("Description")
        rating = st.slider("Rating", 0.0, 5.0, 0.0)
        ingredients = st.text_area("Ingredients (comma separated)")

        if st.button("Add Recipe"):
            ingredients_list = [ingredient.strip() for ingredient in ingredients.split(",")]
            add_recipe(name, description, rating, ingredients_list)

if __name__ == "__main__":
    main()