import streamlit as st
import matplotlib.pyplot as plt
import logging
import requests
import rating_recipe_correlation_analysis as rrca
import pandas as pd


def get_ip():
    """
    Get the public IP address of the user.

    Returns:
        str: The public IP address or an error message.
    """
    try:
        response = requests.get('https://api.ipify.org?format=json')
        ip = response.json()['ip']
        return ip
    except requests.exceptions.RequestException:
        return "Impossible de recevoir l'@IP"


def toggle_dataframe():
    """
    Toggle between DataFrames.

    This function alternates the index of the displayed DataFrame.
    """
    logger.info(
        f"@IP={user_ip} : Appui sur le boutton - Changement de DataFrame")
    st.session_state.df_index = 1 - st.session_state.df_index


def display_fig(fig):
    """
    Display a matplotlib figure and close it to free memory.

    Args:
        fig (Figure): Matplotlib figure to display.
    """
    st.pyplot(fig)
    plt.close()


@st.cache_data
def init_data_part1():
    """
    Load and clean the first dataset.

    Returns:
        DataFrame: Cleaned DataFrame for the first dataset.
    """
    df_cleaned = rrca.create_data_part1()
    logger.info(f"@IP={user_ip} : Chargement du dataset 1")
    return df_cleaned


@st.cache_data
def init_data_part2():
    """
    Load and clean the second dataset.

    Returns:
        DataFrame: Cleaned DataFrame for the second dataset.
    """
    user_analysis_cleaned = rrca.create_data_part2()
    logger.info(f"@IP={user_ip} : Chargement du dataset 2")
    return user_analysis_cleaned


# Configuration du logger pour écrire les logs
logging.basicConfig(level=logging.INFO,
                    format='INFO - [%(asctime)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
user_ip = get_ip()  # Récupérer l'adresse IP de l'utilisateur


def main():
    """
    Main function to analyze recipe data.

    This function initializes and cleans the data, performs various analyses,
    and prints the results. It includes distribution plots, correlation matrices,
    quartile calculations, and linear regression analyses.

    Returns:
        None
    """
    st.title("Analyse des mauvaises recettes")  # Titre de l'application
    df_cleaned = init_data_part1()  # Charger les données du premier JDD
    user_analysis_cleaned = init_data_part2()  # Charger les données du deuxième JDD
    st.sidebar.title("Navigation")  # Titre de la sidebar
    choice = st.sidebar.radio(
        "Allez à :",
        [
            "Introduction",
            "Préparation des datasets"
            "Analyse préliminaire",
            "Influence du temps de préparation et de la complexité",
            "Influence du contenu nutritionnel",
            "Influence de popularité et de la visibilité",
            "Influence des tags et des descriptions",
            "Influence du temps de préparation par étape",
            "Analyse des profils utilisateurs et des biais"
        ]
    )  # Options de la sidebar

    if 'df_index' not in st.session_state:
        st.session_state.df_index = 0  # Initialisation pour afficher df1 au départ
    # Affichage du bouton pour alterner
    st.sidebar.button('Changer de DataFrame', on_click=toggle_dataframe)

    if st.session_state.df_index == 0:  # Affichage du DataFrame sélectionné en fonction de l'état
        st.sidebar.write(f"Le DataFrame {
                         st.session_state.df_index+1} est sélectionné, c'est à dire celui avec les notes moyennes par recettes")
        data = df_cleaned
    else:
        st.sidebar.write(f"Le DataFrame {
                         st.session_state.df_index+1} est sélectionné, c'est à dire celui avec toutes les notes par recettes")
        data = user_analysis_cleaned

# Page 1
    if choice == "Introduction":
        logger.info(f"@IP={user_ip} : Navigation - Introduction")
        st.subheader("Introduction")
        st.write(
            "Bienvenu sur notre application qui permet d'analyser les mauvaises recettes.")
        st.subheader("Auteurs")
        st.write("- Aude De Fornel")
        st.write("- Camille Ishac")
        st.write("- Romain Donné")
        st.write("Lien du GitHub : https://github.com/romainDonne22/cookproject")

# Page
    if choice == "Préparation des datasets":
        logger.info(f"@IP={user_ip} : Navigation - Préparation des datasets")

        # st.subheader("Préparation des datasets")
        # st.write("Affichons des 5 premières lignes de notre JDD : ")
        # st.dataframe(data.head()) # Afficher les 5 premières lignes du tableau pré-traité
        # nb_doublon=rrca.check_duplicates(data) # Vérifier les doublons
        # st.write(f"Nombre de doublons : {nb_doublon}")
        # st.write("Les outliers peuvent grandement affecter les corrélations. Nous les avons supprimés pour cette analyse.")
        # st.write(f"Taille du JDD après suppression des outliers : {data.shape}")
        # st.write("Bienvenu sur notre application qui permet d'analyser les mauvaises recettes.")
        # st.subheader("Auteurs")
        # st.write("- Aude De Fornel")
        # st.write("- Camille Ishac")
        # st.write("- Romain Donné")
        # st.write("Lien du GitHub : https://github.com/romainDonne22/cookproject")

        # Titre principal
        st.subheader("Préparation des Données pour l'Analyse")

        # Description des datasets RAW
        st.markdown("""
        Nous avons travaillé à partir de deux datasets bruts (**RAW**) :
        - **Contributions et leurs informations** : contenant les détails des recettes.
        - **Notes et reviews des utilisateurs** : incluant les évaluations et commentaires.
        """)

        # Sous-section : Construction des datasets
        st.subheader("Construction des Datasets")
        st.markdown("""
        Pour l'analyse, nous avons créé trois datasets distincts :

        1. **Dataset 1** : Concaténation des données nettoyées sur les recettes avec des statistiques sur les notes.
        - Contenu : note moyenne, note médiane, nombre d’utilisateurs ayant noté, notes maximale et minimale, quartiles des notes, nombre de notes égales ou inférieures à 5.

        2. **Dataset 2** : Fusion des données nettoyées sur les recettes avec celles des interactions utilisateur.
        - Contenu : une ligne par note par recette, permettant une analyse détaillée des évaluations individuelles.

        3. **Dataset 3** : Basé sur le Dataset 2, il permet d'étudier les biais utilisateur.
        - Contenu : agrégation par utilisateur incluant le nombre de recettes notées, moyenne, médiane, notes maximale et minimale, et variance des évaluations.
        """)

        # Sous-section : Nettoyage des données
        st.subheader("Nettoyage des Données")
        st.markdown("""
        Les étapes suivantes ont été appliquées à chaque dataset :

        - **Gestion des valeurs manquantes** :
        - Dataset "recipe" : 4979 descriptions manquantes remplacées par "missing".
        - Dataset "users" : 169 reviews manquantes remplacées également par "missing".
        - **Doublons** : Aucune duplication détectée.
        - **Renommage des colonnes** : Conversion en formats conventionnels (ex. : `total fat (%)` → `total_fat`).
        - **Conversion des formats** :
        - IDs transformés en catégories.
        - Variables temporelles (`year`, `month`, `day`) converties en catégories.
        - Dates (`submitted`) mises au format date.
        - **Traitement des valeurs aberrantes** :
        - Seuils établis entre le 15ᵉ et le 85ᵉ percentile.
        - **Cas particulier de la variable `minutes`** :
            - Suppression des valeurs anormalement élevées.
            - Remplacement des `0 minutes` par une valeur aléatoire comprise entre 1 et 15 pour les recettes taguées "15 minutes or less".
        """)
        st.dataframe(
            data.head())  # Afficher les 5 premières lignes du tableau pré-traité
        # nb_doublon=rrca.check_duplicates(data) # Vérifier les doublons
        # st.write(f"Nombre de doublons : {nb_doublon}")
        st.write(
            "Les outliers peuvent grandement affecter les corrélations. Nous les avons supprimés pour cette analyse.")
        st.write(f"Taille du JDD après suppression des outliers : {
                 data.shape}")

        # Détails sur les tailles des datasets
        st.markdown("""
        #### Taille des datasets après nettoyage :
        - **Dataset 1** :
        - Avant : (231,631 lignes, 32 colonnes).
        - Après : (195,230 lignes, 32 colonnes).
        - **Dataset 2** :
        - Avant : (1,132,333 lignes, 24 colonnes).
        - Après : (953,938 lignes, 24 colonnes).
        """)

        # Sous-section : Feature Engineering
        st.subheader("Feature Engineering")
        st.markdown("""
        ##### Dataset 1 :
        - Extraction des contenus nutritionnels (variable `nutrition`) en colonnes distinctes.
        - Décomposition de la variable `submitted` pour obtenir l’année, le mois, le jour et le jour de la semaine, afin d’identifier une éventuelle saisonnalité et son impact sur les évaluations.

        ##### Dataset 2 :
        - Même approche que pour le Dataset 1 concernant les contenus nutritionnels et les informations temporelles.
        - Observation de la distribution de la variable `rating`, asymétrique à gauche.

        ##### Limites identifiées :
        - Absence des données sur les quantités d’ingrédients et le nombre de portions dans le dataset "recipe". Aucune variable exploitable n’a permis de pallier ce manque.
        """)

        # Footer
        st.markdown("---")
        st.markdown("© 2024 - Analyse des Recettes et Notes des Utilisateurs")

# Page 2
    elif choice == "Analyse préliminaire":
        logger.info(f"@IP={user_ip} : Navigation - Analyse préliminaire")
        st.subheader("Analyse préliminaire des corrélations")
        st.write("Nous allons commencer par une analyse préliminaire des corrélations entre les variables numériques et les notes moyennes.")
        # st.write("Affichons des 5 premières lignes de notre JDD : ")
        # st.dataframe(data.head()) # Afficher les 5 premières lignes du tableau pré-traité
        # nb_doublon=rrca.check_duplicates(data) # Vérifier les doublons
        # st.write(f"Nombre de doublons : {nb_doublon}")
        # st.write("Les outliers peuvent grandement affecter les corrélations. Nous les avons supprimés pour cette analyse.")
        # st.write(f"Taille du JDD après suppression des outliers : {data.shape}")

        if st.session_state.df_index == 0:
            # Distibution de la moyenne des notes
            st.write("Distribution de la moyenne des notes : ")
            display_fig(rrca.plot_distribution(
                data, 'note_moyenne', 'Distribution de la moyenne'))
            # Distibution de la médiane des notes
            st.write("Distrubution de la médiane des notes : ")
            display_fig(rrca.plot_distribution(
                data, 'note_mediane', 'Distribution de la médiane'))
            st.write("Nous n’avons trouvé aucune corrélation initiale. Nous supposons que l’utilisation de la moyenne a pu masquer les corrélations. Pour vérifier cela, nous avons testé les corrélations entre les variables des recettes et les notes des utilisateurs directement. Pour observer les changements, cliquez sur le bouton ‘Changer de DataFrame’")

        else:
            # Distibution de la moyenne des notes
            st.write("Distribution des notes : ")
            display_fig(rrca.plot_distribution(
                data, 'rating', 'Distribution de la moyenne'))

        # st.subheader("Qu'est-ce qui caractérise une mauvaise recette ? : ")
        # st.write("La première partie de l'analyse portera sur l'analyse des contributions qui ont eu une moyenne de moins de 4/5 ou égale à 4 :")
        # st.write("Quels sont les critères d'une mauvaise recette/contribution ?")
        # st.write("Quelles sont les caractéristiques des recettes les moins populaires ?")
        # st.write("Qu'est-ce qui fait qu'une recette est mal notée?")

        # Matrice de corrélation
        st.write("Regardons la matrice de corrélation et les boxplots :")
        if st.session_state.df_index == 0:
            display_fig(rrca.plot_correlation_matrix(
                data,
                ['note_moyenne', 'minutes', 'n_steps', 'n_ingredients', 'calories', 'total_fat',
                 'sugar', 'sodium', 'protein', 'saturated_fat', 'carbohydrates', 'nb_user'],
                "Matrice de corrélation entre la moyenne des notes et les autres variables numériques"
            ))
            st.write(
                "Nous allons commencer par une analyse préliminaire des corrélations entre les variables numériques et les notes moyennes.")

        else:
            display_fig(rrca.plot_correlation_matrix(
                data,
                ['rating', 'minutes', 'n_steps', 'n_ingredients', 'calories', 'total_fat', 'sugar',
                 'sodium', 'protein', 'carbohydrates', 'binary_rating'],
                "Matrice de corrélation entre les notes et les autres variables numériques"
            ))
            st.write(
                "Nous n’avons trouvé aucune corrélation avec la variable rating ou la variable binary_rating")

        # st.write("Pas de corrélation entre les notes et les variables sélectionnées dans la matrice de correlation (hormis avec binary_rating, variable qui est construite à partir de rating et qui nous sert à savoir si la note est supérieure ou égale à 4 ou non).")

        # Boxplot
        numerical_cols = data.select_dtypes(
            include=['int64', 'float64']).columns
        for colonne in numerical_cols:
            display_fig(rrca.boxplot_numerical_cols(data, colonne))

        st.write("Toujours pas de corrélations avec notre variable note_moyenne. Il se peut que le passage à la moyenne altère les corrélations.",
                 "Continuous l'analyse en comparant des metrics pour les good et bad ratings, nous reviendrons à ce problème de moyenne dans un deuxième temps.")
        st.write("Regardons à quelle note correspond le 1e quartile. Nous nous concentrerons sur les 25% moins bonnes recettes pour notre analyse.")

        if st.session_state.df_index == 0:
            # Calcul des quartiles
            mean_quartile = rrca.calculate_quartile(data, 'note_moyenne', 0.25)
            st.write("3e Quartile pour la moyenne:", mean_quartile)
            mean_quartile = rrca.calculate_quartile(data, 'note_mediane', 0.25)
            st.write("3e Quartile pour la médiane:", mean_quartile)
            # Nombre de mauvaises notes
            st.write(f"Nombre de recettes avec une moyenne inférieure à 4 : {
                     data[data['note_moyenne'] <= 4.0].shape[0]}")
            st.write(f"Nombre de recettes avec une médiane inférieure à 4 : {
                     data[data['note_mediane'] <= 4.0].shape[0]}")
            st.write("Nous nous concentrerons sur la moyenne qui nous permet d'augmenter l'échantillon de bad ratings. Compte tenu de la distribution de la moyenne, on peut considérer les 4 (et moins) comme des mauvaises notes.")
            # Séparer les recettes mal notées des bien notées
            bad_ratings, good_ratings = rrca.separate_bad_good_ratings(
                data, 4, 'note_moyenne')

        else:
            # Calcul des quartiles
            mean_quartile = rrca.calculate_quartile(data, 'rating', 0.25)
            st.write("3e Quartile pour la note:", mean_quartile)
            # Nombre de mauvaises notes
            st.write(f"Nombre de recettes avec une note inférieure à 4 : {
                     data[data['rating'] <= 4.0].shape[0]}")
            st.write("Nous nous concentrerons sur la note qui nous permet d'augmenter l'échantillon de bad ratings. Compte tenu de la distribution de la note, on peut considérer les 4 (et moins) comme des mauvaises notes.")
            # Séparer les recettes mal notées des bien notées
            bad_ratings, good_ratings = rrca.separate_bad_good_ratings(
                data, 4, 'rating')

        # Filtrer les recettes avec une note inférieure ou égale à 4 :
        st.write("Afin de comparer les recettes mal notées des bien notées, nous devons filtrer le dataframe sur les mauvaises notes (première ligne) et les bonnes notes (deuxième ligne). ")
        display_fig(rrca.plot_bad_ratings_distributions(
            bad_ratings, good_ratings))
        st.write("Pas de grosses variations à observer... Regardons maintenant si la saisonnalité / la période où la recette est postée a un impact :)")

        # Saisonalité
        st.write(
            "Saisonalié des recettes mal notées (en haut) et bien notées (en bas) : ")
        display_fig(rrca.saisonnalite(bad_ratings))
        display_fig(rrca.saisonnalite(good_ratings))
        st.write(
            "Nous n'observons pas d'impact de la saisonnalité du post entre bad et good ratings.")

# Page 3
    elif choice == "Influence du temps de préparation et de la complexité":
        logger.info(
            f"@IP={user_ip} : Navigation - Influence du temps de préparation et de la complexité")
        # Comparaison du temps, du nombre d'étapes et du nombre d'ingrédients entre les recettes bien et mal notées
        st.subheader(
            "Analyser l'impact du temps de préparation and la complexité sur les notes :")

        if st.session_state.df_index == 0:
            bad_ratings, good_ratings = rrca.separate_bad_good_ratings(
                data, 4, 'note_moyenne')
        else:
            bad_ratings, good_ratings = rrca.separate_bad_good_ratings(
                data, 4, 'rating')

        data_minutes = [good_ratings['minutes'], bad_ratings['minutes']]
        display_fig(rrca.boxplot_df(data_minutes))
        data_steps = [good_ratings['n_steps'], bad_ratings['n_steps']]
        display_fig(rrca.boxplot_df(data_steps))
        data_ingred = [good_ratings['n_ingredients'],
                       bad_ratings['n_ingredients']]
        display_fig(rrca.boxplot_df(data_ingred))
        st.write("Les recettes mal notées tendent à avoir des temps de préparation plus longs et un nombre d'étapes à suivre plus élevé. Rien à signalier sur le nombre d'ingrédients.")

        # Distribution de la note par rapport à la variable minutes / n_steps / n_ingredients en %:
        st.write("Pour aller plus loin dans l'analyse nous allons créer des bins pour chaque variable avec des seuils définis (low, medium, high) et regarder la proportion des moyennes dans chaque catégorie.")

        if st.session_state.df_index == 0:
            fig, comparison_minutes = rrca.rating_distribution(
                df=data,
                variable='minutes',
                rating_var='note_moyenne',
                low_threshold=15,
                mean_range=(30, 50),
                high_threshold=180
            )
            display_fig(fig)
            st.write(
                "Distribution de la note par rapport à la variable minutes en %:")
            st.write(comparison_minutes)

            fig, comparison_steps = rrca.rating_distribution(
                df=data,
                variable='n_steps',
                rating_var='note_moyenne',
                low_threshold=3,
                mean_range=(8, 10),
                high_threshold=15
            )
            display_fig(fig)
            st.write(
                "Distribution de la note par rapport à la variable n_steps en %:")
            st.write(comparison_steps)

            fig, comparison_ingr = rrca.rating_distribution(
                df=data,
                variable='n_ingredients',
                rating_var='note_moyenne',
                low_threshold=3,
                mean_range=(8, 10),
                high_threshold=15
            )
            display_fig(fig)
        else:
            fig, comparison_minutes = rrca.rating_distribution(
                df=data,
                variable='minutes',
                rating_var='rating',
                low_threshold=15,
                mean_range=(30, 50),
                high_threshold=180
            )
            display_fig(fig)
            st.write(
                "Distribution de la note par rapport à la variable minutes en %:")
            st.write(comparison_minutes)

            fig, comparison_steps = rrca.rating_distribution(
                df=data,
                variable='n_steps',
                rating_var='rating',
                low_threshold=3,
                mean_range=(8, 10),
                high_threshold=15
            )
            display_fig(fig)
            st.write(
                "Distribution de la note par rapport à la variable n_steps en %:")
            st.write(comparison_steps)

            fig, comparison_ingr = rrca.rating_distribution(
                df=data,
                variable='n_ingredients',
                rating_var='rating',
                low_threshold=3,
                mean_range=(8, 10),
                high_threshold=15
            )
            display_fig(fig)

        st.write(
            "Distribution de la note par rapport à la variable n_ingredients en %:")
        st.write(comparison_ingr)
        st.write("Même analyse pour la variable nombre d'étapes : plus les recettes ont un nombre d'étapes élevé / sont complexes plus elles sont mal notées. A contrario les recettes avec moins de 3 étapes sont sensiblement mieux notées.")
        st.write(
            "Le nombre d'ingrédients en revanche ne semble pas impacté la moyenne.")

        st.write("Réalisons une régression avec ces trois variables pour comprendre dans quelle mesure elles impactent la note et si cette hypothèse est statistiquement viable.")
        st.write(
            "La matrice de corrélation en les variables 'minutes','n_steps','n_ingredients' est la suivante")
        columns_to_analyze = ['minutes', 'n_steps', 'n_ingredients']
        correlation = rrca.correlation(data, columns_to_analyze)
        st.write(correlation)

        # Régression linéaire
        st.write(
            "Régression linéaire entre les variables 'minutes','n_steps','n_ingredients' et la note moyenne : ")
        X = data[['minutes', 'n_steps']]

        if st.session_state.df_index == 0:
            y = data['note_moyenne']
        else:
            y = data['rating']

        model = rrca.OLS_regression(X, y)
        st.write(model.summary())
        st.write("ANALYSE :")
        st.write("R-Squared = O.OO1 -> seulement 0.1% de la variance dans les résultats est expliquée par les variables n_steps et minutes. "
                 "C'est très bas, ces variables ne semblent pas avoir de pouvoir prédictif sur les ratings, même si on a pu détecter des tendances de comportements users.")
        st.write("Prob (F-Stat) = p-value est statistiquement signifiante (car < 0.05) -> au moins un estimateur a une relation linéaire avec note_moyenne. "
                 "Cependant l'effet sera minime, comme le montre le résultat R-Squared")
        st.write("Coef minute : VERY small. p-value < 0.05 donc statistiquement signifiant mais son effet est quasi négligeable sur note_moyenne. "
                 "Même constat pour n_steps même si l'effet est légèrement supérieur : une augmentation de 10 étapes va baisser la moyenne d'environ 0.025...")
        st.write("Les tests Omnibus / Prob(Omnibus) et Jarque-Bera (JB) / Prob(JB) nous permettent de voir que les résidus ne suivent probablement pas une distribution gaussienne, les conditions pour une OLS ne sont donc pas remplies.")
        st.write(
            "--> il va falloir utiliser une log transformation pour s'approcher de variables gaussiennes.")

        # 35) Régression linéaire avec log transformation
        data['minutes_log'] = rrca.dflog(data, 'minutes')
        data['n_steps_log'] = rrca.dflog(data, 'n_steps')
        X = data[['minutes_log', 'n_steps_log']]

        if st.session_state.df_index == 0:
            y = data['note_moyenne']
        else:
            y = data['rating']

        model = rrca.OLS_regression(X, y)
        st.write(model.summary())
        st.write("ANALYSE :")
        st.write("En passant au log, on se rend compte que la variable minute a plus de poids sur la moyenne que le nombre d'étapes. Néanmoins bien que les variables minutes_log et n_steps_log soient statistiquement significatives (cf p value), leur contribution à la prédiction de la note moyenne est très faible.")
        st.write("En effet R2 est toujours extrêmement petit donc ces deux variables ont un impact minime sur la moyenne, qui ne permet pas d'expliquer les variations de la moyenne.")
        st.write("Il est probablement nécessaire d'explorer d'autres variables explicatives ou d'utiliser un modèle non linéaire pour mieux comprendre la note_moyenne.")

# Page 4

    elif choice == "Influence du contenu nutritionnel":
        logger.info(
            f"@IP={user_ip} : Navigation - Influence du contenu nutritionnel")
        st.subheader(
            "Analyser le contenu nutritionnel des recettes et leur impact sur les notes")

        # Comparaison calories
        if st.session_state.df_index == 0:
            fig, comparison_calories = rrca.rating_distribution(
                df=data,
                variable='calories',
                rating_var='note_moyenne',
                low_threshold=100,
                mean_range=(250, 350),
                high_threshold=1000
            )
        else:
            fig, comparison_calories = rrca.rating_distribution(
                df=data,
                variable='calories',
                rating_var='rating',
                low_threshold=100,
                mean_range=(250, 350),
                high_threshold=1000
            )
        st.write("\nComparison of Rating Distribution in %:")
        display_fig(fig)
        st.write("Distribution de la note par rapport à la variable calories en %:")
        st.write(comparison_calories)

        # Comparaison total_fat
        if st.session_state.df_index == 0:
            fig, comparison_total_fat = rrca.rating_distribution(
                df=data,
                variable='total_fat',
                rating_var='note_moyenne',
                low_threshold=8,
                mean_range=(15, 25),
                high_threshold=100
            )
        else:
            fig, comparison_total_fat = rrca.rating_distribution(
                df=data,
                variable='total_fat',
                rating_var='rating',
                low_threshold=8,
                mean_range=(15, 25),
                high_threshold=100
            )
        st.write("\nComparison of Rating Distribution in %:")
        display_fig(fig)
        st.write("Distribution de la note par rapport à la variable total_fat en %:")
        st.write(comparison_total_fat)

        # Comparaison sugar
        if st.session_state.df_index == 0:
            fig, comparison_sugar = rrca.rating_distribution(
                df=data,
                variable='sugar',
                rating_var='note_moyenne',
                low_threshold=8,
                mean_range=(15, 25),
                high_threshold=60
            )
        else:
            fig, comparison_sugar = rrca.rating_distribution(
                df=data,
                variable='sugar',
                rating_var='rating',
                low_threshold=8,
                mean_range=(15, 25),
                high_threshold=60
            )
        st.write("\nComparison of Rating Distribution in %:")
        display_fig(fig)
        st.write("Distribution de la note par rapport à la variable sugar en %:")
        st.write(comparison_sugar)

        # Comparaison protein
        if st.session_state.df_index == 0:
            fig, comparison_protein = rrca.rating_distribution(
                df=data,
                variable='protein',
                rating_var='note_moyenne',
                low_threshold=8,
                mean_range=(15, 25),
                high_threshold=60
            )
        else:
            fig, comparison_protein = rrca.rating_distribution(
                df=data,
                variable='protein',
                rating_var='rating',
                low_threshold=8,
                mean_range=(15, 25),
                high_threshold=60
            )
        st.write("\nComparison of Rating Distribution in %:")
        display_fig(fig)
        st.write("Distribution de la note par rapport à la variable protein en %:")
        st.write(comparison_protein)

        # Conclusion
        st.write(
            "Les variations sont trop faibles. Les contenus nutritionnels des recettes n'impactent pas la moyenne.")

# Page 5
    elif choice == "Influence de popularité et de la visibilité":
        logger.info(
            f"@IP={user_ip} : Navigation - Influence de popularité et de la visibilité")
        st.subheader(
            "Analyser l'impact de la popularité et de la visibilité des recettes sur les notes")

        if st.session_state.df_index == 0:
            # Calculer Q1, Q3, et IQR pour le nb_users
            Q1_nb_user = rrca.calculate_quartile(data, 'nb_user', 0.25)
            Q2_nb_user = rrca.calculate_quartile(data, 'nb_user', 0.50)
            Q3_nb_user = rrca.calculate_quartile(data, 'nb_user', 0.75)
            st.write("Q1 pour le nombre d'utilisateurs : ", Q1_nb_user)
            st.write("Q2 pour le nombre d'utilisateurs : ", Q2_nb_user)
            st.write("Q3 pour le nombre d'utilisateurs : ", Q3_nb_user)

            # Comparaison popularity
            fig, comparison_popularity = rrca.rating_distribution(
                df=data,
                variable='nb_user',
                rating_var='note_moyenne',
                low_threshold=2,
                mean_range=(2, 3),
                high_threshold=4
            )
            st.write("\nComparison of Rating Distribution in %:")
            display_fig(fig)
            st.write(
                "Distribution de la note par rapport à la variable popularity en %:")
            st.write(comparison_popularity)

            # Conclusion
            st.write("Il est très net ici que les recettes ayant le moins de notes sont celles les moins bien notées. "
                     "Cela veut dire qu'elles sont moins populaires et/ou moins visibles. Au contraire celles avec le plus de notes sont les mieux notées.")
            st.write("Ou ça peut vouloir dire que les utilisateurs ne notent pas les mauvaises recettes. La mauvaise note appelle la mauvaise note.")
            st.write("A CREUSER :")
            st.write("- qui sont les users qui ont mal noté ces recettes : ont-ils beaucoup noté ? Mettent-ils que des mauvaises notes ? Pour vérifier si cette information est significative.")
            st.write("- faire un heatmap : nb_users/note_moyenne")
        else:
            st.write("Il ne sert à rien de passer sur le data set 2 car pour cette partie car nous traçons nb_users en fonction de la note moyenne. Merci donc de revenir sur le data set 1 pour cette analyse.")

# Page 6
    elif choice == "Influence des tags et des descriptions":
        logger.info(
            f"@IP={user_ip} : Navigation - Influence des tags et des descriptions")
        st.subheader(
            "Analyses des variables categorical - tags & descriptions - pour comprendre grâce au verbage les critères d'une mauvaise note")

        if st.session_state.df_index == 0:
            bad_ratings, good_ratings = rrca.separate_bad_good_ratings(
                data, 4, 'note_moyenne')

            st.write("Analysons les tags et descriptions pour essayer de trouver des thèmes communs entre les recettes mal notées. "
                     "On les comparera aux recettes bien notées. Pour cela nous utiliserons les dataframes bad_ratings et good_ratings. "
                     "La première étape est de réaliser un pre-processing de ces variables (enlever les mots inutiles, tokeniser).")

            # Preprocessing des tags et descriptions
            bad_ratings.loc[:, 'tags_clean'] = bad_ratings.loc[:,
                                                               'tags'].fillna('').apply(rrca.preprocess_text)
            bad_ratings.loc[:, 'description_clean'] = bad_ratings.loc[:,
                                                                      'description'].fillna('').apply(rrca.preprocess_text)
            good_ratings.loc[:, 'tags_clean'] = good_ratings.loc[:, 'tags'].fillna(
                '').apply(rrca.preprocess_text)
            good_ratings.loc[:, 'description_clean'] = good_ratings.loc[:,
                                                                        'description'].fillna('').apply(rrca.preprocess_text)

            # Mots les plus courants dans les tags des recettes mal notées
            most_common_bad_tags_clean = rrca.get_most_common_words(
                bad_ratings['tags_clean'])
            st.write("Les tags les plus courants dans les recettes mal notées :")
            bad_tag_words_set = rrca.extractWordFromTUpple(
                most_common_bad_tags_clean)
            st.write(bad_tag_words_set)

            # Mots les plus courants dans la descriptions des recettes mal notées
            most_common_bad_desciption_clean = rrca.get_most_common_words(
                bad_ratings['description_clean'])
            st.write(
                "\nLes mots les plus courants dans les descriptions des recettes mal notées :")
            bad_desc_words_set = rrca.extractWordFromTUpple(
                most_common_bad_desciption_clean)
            st.write(bad_desc_words_set)

            # Mots les plus courants dans les tags des recettes bien notées
            most_common_good_tags_clean = rrca.get_most_common_words(
                good_ratings['tags_clean'])
            st.write("Les tags les plus courants dans les recettes bien notées :")
            good_tag_words_set = rrca.extractWordFromTUpple(
                most_common_good_tags_clean)
            st.write(good_tag_words_set)

            # Mots les plus courants dans descriptions des recettes bien notées
            most_common_good_desciption_clean = rrca.get_most_common_words(
                good_ratings['description_clean'])
            st.write(
                "\nLes mots les plus courants dans les descriptions des recettes bien notées :")
            good_desc_words_set = rrca.extractWordFromTUpple(
                most_common_good_desciption_clean)
            st.write(good_desc_words_set)

            # Mots uniques dans les tags et descriptions des recettes mal notées :
            st.write("Mots uniques dans les tags des recettes mal notées :",
                     rrca.uniqueTags(bad_tag_words_set, good_tag_words_set))
            st.write("Mots uniques dans les descriptions des recettes mal notées :",
                     rrca.uniqueTags(bad_desc_words_set, good_desc_words_set))
        else:
            # la fonction marche mais en local uniquement, trop lourde en RAM pour le serveur
            bad_ratings, good_ratings = rrca.separate_bad_good_ratings(
                data, 4, 'rating')
            st.write(
                "Sur le dataset 2, les calculs prennent bcp trop de temps et d'espace RAM ce qui provoquait des crashs.")
            st.write(
                "Nous avons donc fait tourner en local les calculs et les résultats sont les suivants :")
            st.write(
                "Mots uniques dans les tags des recettes mal notées : {'rice', 'fish'}")
            st.write(
                "Mots uniques dans les tags des recettes mal notées : {'low', 'healthy'}")

        # Conclusion
        st.write(
            "Il vaut mieux éviter d'écrire une recette avec les mots et les descriptions ci-dessus.")
        st.write("Notons que les résultats sont différents entre les deux datasets.")

# Page 7
    elif choice == "Influence du temps de préparation par étape":
        logger.info(
            f"@IP={user_ip} : Navigation - Influence du temps de préparation par étape")
        st.write(
            "Pour cette étude on va se concentrer sur le temps moyen par étape et non le temps total de préparation.")

        if st.session_state.df_index == 0:
            st.write(
                "La fonction permettant de calculer le temps moyen par étape pour chaque recette ne peut fonctionner que sur le dataset 2.")
            st.write("Merci donc de changer de dataset.")
        else:
            fig = rrca.time_per_step(data, 'minutes', 'n_steps')
            display_fig(fig)
            st.write(
                "Plus le rapport temps par étape est élevé plus la proportion de recettes mal notée augmente.")
            st.write("Nous en concluons que l'incohérence entre la durée totale d’une recette et la longueur du processus de préparation génère des mauvaises notes, gage d'un manque de qualité de ces contributions.")
            st.write(
                "Vérifions s'il existe une relation linéaire entre les notes et cette variables :")
            # Régression linéaire
            data['minute_step'] = data['minutes'] / data['n_steps']
            X = data['minute_step']
            y = data['rating']
            model = rrca.OLS_regression(X, y)
            st.write(model.summary())
            st.write(
                "Le modèle linéaire n'est pas applicable ici, R2 étant proche de 0.")

# Page 8
    elif choice == "Analyse des profils utilisateurs et des biais":
        logger.info(
            f"@IP={user_ip} : Navigation - Analyse des profils utilisateurs et des biais")
        st.write("Pour cette étude nous analysons les profils des utilisateurs (contributeurs vs non-contributeurs) et leur impact sur les notes.")

        if st.session_state.df_index == 0:
            st.write(
                "La fonction qui définit les profils utilisateurs et leurs caractéristiques se base sur le dataset 2.")
            st.write("Merci de changer de dataset pour consulter l'analyse.")
        else:
            st.write("Jusqu’ici, notre analyse a montré qu’il n'existe pas de modèles linéaires pour expliquer les notes du point de vue de la qualité des contributions. C’est pourquoi nous nous sommes demandé si les notes ne seraient pas biaisées. Nous allons analyser dans cette partie les biais des utilisateurs : ")
            st.write("  - Existe-t-il une corrélation entre le nombre de recettes notées par utilisateur et la moyenne des notes qu’ils attribuent ou leur dispersion ? ")
            st.write("  - Les utilisateurs les plus contributeurs sont-ils ceux qui obtiennent les meilleures notes ? Notent-ils les plus sévèrement ? ")
            st.write(
                "  - Les contributeurs évaluent-ils davantage de recettes que les non-contributeurs ? ")
            st.write(
                "  - Les utilisateurs qui notent beaucoup de recettes sont-ils aussi ceux qui donnent les notes les plus extrêmes ? ")
            st.write(
                "Autant de questions que nous allons explorer dans cette partie de notre analyse.")
            st.write("")
            st.write("Les premières lignes du dataset analysé : ")
            user_profiles = rrca.create_dfuser_profiles(data)
            st.dataframe(user_profiles.head())

            # Distribution du nombre de raters également contributeurs
            st.write(
                "Les contributeurs évaluent-ils davantage de recettes que les non-contributeurs ?")
            fig = rrca.rating_isContributor(user_profiles, 'is_contributor')
            display_fig(fig)

            # Moyenne du nombre de recettes notées pour contributeurs et non-contributeurs
            contributor_stats = user_profiles.groupby(
                'is_contributor')['num_recipes_rated'].mean()
            st.write(
                "Moyenne du nombre de recettes notées par les contributeurs et les non-contributeurs :")
            st.write(contributor_stats)
            st.write("Les contributeurs notent en moyenne 19 fois plus de recettes que les non-contributeurs. Pourtant ils sont largement moins nombreux parmi les utilisateurs qui notent.")
            st.write(
                "Explorons quels types utilisateurs donnent systématiquement des notes basses ou élevées.")
            st.write("")
            # Moyenne et variance des notes par profil utilisateur
            data = {
                'is_contributor': [False, True],
                'mean_rating': [3.871002, 4.411131],
                'var_rating': [0.366178, 0.997883],
            }
            # Création du DataFrame
            df = pd.DataFrame(data).set_index('is_contributor')
            # Afficher le tableau avec Streamlit
            st.write(
                "Statistiques des notes pour contributeurs et non-contributeurs")
            st.dataframe(df.style.format(
                {"mean_rating": "{:.6f}", "var_rating": "{:.6f}"}))
            # Distribution des notes moyennes par groupe (contributeur vs non-contributeur)
            st.write(
                "Distribution des notes moyennes par groupe (contributeur vs non-contributeur)")
            fig = rrca.plot_distributionIsContributor(
                user_profiles, 'is_contributor', 'mean_rating')
            display_fig(fig)
            st.write("Les utilisateurs non-contributeurs sont ceux qui notent les plus sévèremment. Ils sont également les plus homogènes dans leur notation.")
            st.write("Les contributeurs notent plus de recettes et ils les notent bien. Cependant ils sont beaucoup plus dispersés dans leur notation. Ceci constitue un premier biais qui tire les notes et les moyennes vers le haut.")
            st.write("Les contributeurs sont des ambassadeurs du site alors que les utilisateurs occasionnels ne semblent noter des recettes que lorsqu'ils sont insatisfaits. Cela constitue une autre forme de biais dans les notes.")
            user_profiles = None  # Libérer la mémoire


if __name__ == "__main__":
    main()
