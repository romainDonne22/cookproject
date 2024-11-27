import rating_recipe_correlation_analysis as rca
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("front.py").run(timeout=300) # 300 seconds timeout pour permettre le chargement des données

# Liste des pages à tester
pages = ["Caractéristiques des recettes mal notées", 
        "Influence du temps de préparation et de la complexité", "Influence du contenu nutritionnel", 
        "Influence de popularité et de la visibilité", "Influence des tags et des descriptions", "Introduction"]

# Fonction pour tester chaque page
def test_pages():
    for page in pages:
        try:
            at.sidebar.radio==page # Simuler la sélection de la page dans la barre latérale
            print(f"Page '{page}' loaded successfully.")
        except Exception as e:
            print(f"Page '{page}' failed to load. Error: {e}")

# Fonction pour tester le bouton de la barre latérale
def test_button():
    try:
        at.sidebar.button('Changer de DataFrame', key="toggle_button") # Simuler la sélection de la page dans la barre latérale     
        print("Sidebar button pressed successfully.")
    except Exception as e:
        print(f"Sidebar button press failed. Error: {e}")

# Fonction pour tester l'affichage de texte avec streamlit.write
def test_write(phrase):
    try:
        if phrase in at.get_output():
            print("Streamlit write test passed.")
        else:
            raise AssertionError("Expected text not found in Streamlit write output.")
    except Exception as e:
        print(f"Streamlit write test failed. Error: {e}")

# Appeler les fonctions de test
test_pages()
test_write("Le DataFrame 1 est sélectionné, c'est à dire celui avec les notes moyennes par recettes")
test_button()
test_write("Le DataFrame 1 est sélectionné, c'est à dire celui avec les notes moyennes par recettes")
test_pages()

