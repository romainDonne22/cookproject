import streamlit as st
import matplotlib.pyplot as plt
import logging
import rating_recipe_correlation_analysis as rrca

# Configuration du logger pour écrire les logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
  
# Fonction pour alterner entre les DataFrames
def toggle_dataframe():
    st.session_state.df_index = 1 - st.session_state.df_index # On alterne l'index du DataFrame affiché

# Fonction pour afficher les figures et les fermer après affichage afin de libérer la mémoire
def display_fig(fig):
    st.pyplot(fig)
    plt.close()

# Charger le premier JDD (notes moyennées) une seule fois en cache (c'est le @) sur le serveur Streamlit Hub
@st.cache_data 
def init_data_part1():
    df_cleaned = rrca.create_data_part1()
    return df_cleaned

# Charger le deuxième JDD (toutes les notes) une seule fois en cache (c'est le @) sur le serveur Streamlit Hub
@st.cache_data
def init_data_part2():
    user_analysis_cleaned = rrca.create_data_part2()
    return user_analysis_cleaned


def main():
    st.title("Analyse des mauvaises recettes") # Titre de l'application
    df_cleaned = init_data_part1() # Charger les données du premier JDD
    user_analysis_cleaned = init_data_part2() # Charger les données du deuxième JDD
    st.sidebar.title("Navigation") # Titre de la sidebar
    choice = st.sidebar.radio("Allez à :", ["Introduction", "Caractéristiques des recettes mal notées", 
        "Influence du temps de préparation et de la complexité", "Influence du contenu nutritionnel", 
        "Influence de popularité et de la visibilité", "Influence des tags et des descriptions", 
        "Influence du temps par étape", "Analyse des profils utilisateurs"]) # Options de la sidebar
    if 'df_index' not in st.session_state:
        st.session_state.df_index = 0  # Initialisation pour afficher df1 au départ
    st.sidebar.button('Changer de DataFrame', on_click=toggle_dataframe) # Affichage du bouton pour alterner
    if st.session_state.df_index == 0: # Affichage du DataFrame sélectionné en fonction de l'état
        st.sidebar.write(f"Le DataFrame {st.session_state.df_index+1} est sélectionné, c'est à dire celui avec les notes moyennes par recettes")
        data = df_cleaned
    else:
        st.sidebar.write(f"Le DataFrame {st.session_state.df_index+1} est sélectionné, c'est à dire celui avec toutes les notes par recettes")
        data = user_analysis_cleaned
    
###### Page 1
    if choice == "Introduction":
        logger.info("Naviguation - Introduction")
        st.subheader("Introduction")
        st.write("Bienvenu sur notre application qui permet d'analyser les mauvaises recettes.")
        st.subheader("Auteurs")
        st.write("- Aude De Fornel")
        st.write("- Camille Ishac")
        st.write("- Romain Donné")
        st.write("Lien du GitHub : https://github.com/romainDonne22/cookproject")

###### Page 2
    elif choice == "Caractéristiques des recettes mal notées":
        logger.info("Naviguation - Caractéristiques des recettes mal notées")
        st.subheader("Qu'est-ce qui caractérise une mauvaise recette ?")
        st.write("Affichons des 5 premières lignes de notre JDD : ")
        st.dataframe(data.head()) # Afficher les 5 premières lignes du tableau pré-traité
        nb_doublon=rrca.check_duplicates(data) # Vérifier les doublons
        st.write(f"Nombre de doublons : {nb_doublon}")
        st.write("Les outliers peuvent grandement affecter les corrélations. Nous les avons supprimés pour cette analyse.")
        st.write(f"Taille du JDD après suppression des outliers : {data.shape}")
        
        if st.session_state.df_index == 0 :
            # Distibution de la moyenne des notes
            st.write("Distribution de la moyenne des notes : ")
            display_fig(rrca.plot_distribution(data, 'note_moyenne', 'Distribution de la moyenne'))
            # Distibution de la médiane des notes
            st.write("Distrubution de la médiane des notes : ")
            display_fig(rrca.plot_distribution(data, 'note_mediane', 'Distribution de la médiane'))
            
        else:
            # Distibution de la moyenne des notes
            st.write("Distribution des notes : ")
            display_fig(rrca.plot_distribution(data, 'rating', 'Distribution de la moyenne'))

        st.subheader("Qu'est-ce qui caractérise une mauvaise recette ? : ")
        st.write("La première partie de l'analyse portera sur l'analyse des contributions qui ont eu une moyenne de moins de 4/5 ou égale à 4 :")
        st.write("Quels sont les critères d'une mauvaise recette/contribution ?")
        st.write("Quelles sont les caractéristiques des recettes les moins populaires ?")
        st.write("Qu'est-ce qui fait qu'une recette est mal notée?")

        # Matrice de corrélation
        st.write("Regardons la matrice de corrélation et les boxplots :")
        if st.session_state.df_index == 0 :
            display_fig(rrca.plot_correlation_matrix(data, ['note_moyenne', 'minutes', 'n_steps', 'n_ingredients', 'calories', 'total_fat', 
                            'sugar', 'sodium','protein', 'saturated_fat', 'carbohydrates', 'nb_user'], 
                            "Matrice de corrélation entre la moyenne des notes et les autres variables numériques"))
        else :
            display_fig(rrca.plot_correlation_matrix(data,['rating', 'minutes', 'n_steps', 'n_ingredients', 
                            'calories', 'total_fat', 'sugar', 'sodium', 'protein', 'carbohydrates','binary_rating'], "Matrice de corrélation entre les notes et les autres variables numériques"))
        st.write("Pas de corrélation entre les notes et les variables sélectionnées dans la matrice de correlation (hormis avec binary_rating, variable qui est construite à partir de rating et qui nous sert à savoir si la note est supérieure ou égale à 4 ou non).")
        
        # Boxplot
        numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
        for colonne in numerical_cols:
            display_fig(rrca.boxplot_numerical_cols(data, colonne))
             
        st.write("Toujours pas de corrélations avec notre variable note_moyenne. Il se peut que le passage à la moyenne altère les corrélations.",
                 "Continuous l'analyse en comparant des metrics pour les good et bad ratings, nous reviendrons à ce problème de moyenne dans un deuxième temps.")
        st.write("Regardons à quelle note correspond le 1e quartile. Nous nous concentrerons sur les 25% moins bonnes recettes pour notre analyse.")

        if st.session_state.df_index == 0 :
            # Calcul des quartiles
            mean_quartile=rrca.calculate_quartile(data, 'note_moyenne', 0.25)
            st.write("3e Quartile pour la moyenne:", mean_quartile)
            mean_quartile=rrca.calculate_quartile(data, 'note_mediane', 0.25)
            st.write("3e Quartile pour la médiane:", mean_quartile)
            # Nombre de mauvaises notes
            st.write(f"Nombre de recettes avec une moyenne inférieure à 4 : {data[data['note_moyenne'] <= 4.0].shape[0]}")
            st.write(f"Nombre de recettes avec une médiane inférieure à 4 : {data[data['note_mediane'] <= 4.0].shape[0]}")
            st.write("Nous nous concentrerons sur la moyenne qui nous permet d'augmenter l'échantillon de bad ratings. Compte tenu de la distribution de la moyenne, on peut considérer les 4 (et moins) comme des mauvaises notes.")
            # Séparer les recettes mal notées des bien notées
            bad_ratings, good_ratings = rrca.separate_bad_good_ratings(data, 4, 'note_moyenne')

        else :
            # Calcul des quartiles
            mean_quartile=rrca.calculate_quartile(data, 'rating', 0.25)
            st.write("3e Quartile pour la note:", mean_quartile)
            # Nombre de mauvaises notes
            st.write(f"Nombre de recettes avec une note inférieure à 4 : {data[data['rating'] <= 4.0].shape[0]}")
            st.write("Nous nous concentrerons sur la note qui nous permet d'augmenter l'échantillon de bad ratings. Compte tenu de la distribution de la note, on peut considérer les 4 (et moins) comme des mauvaises notes.")
            # Séparer les recettes mal notées des bien notées
            bad_ratings, good_ratings = rrca.separate_bad_good_ratings(data, 4, 'rating')

        # Filtrer les recettes avec une note inférieure ou égale à 4 :
        st.write("Afin de comparer les recettes mal notées des bien notées, nous devons filtrer le dataframe sur les mauvaises notes (première ligne) et les bonnes notes (deuxième ligne). ")
        display_fig(rrca.plot_bad_ratings_distributions(bad_ratings, good_ratings))
        st.write("Pas de grosses variations à observer... Regardons maintenant si la saisonnalité / la période où la recette est postée a un impact :)")

        # Saisonalité
        st.write("Saisonalié des recettes mal notées (en haut) et bien notées (en bas) : ")
        display_fig(rrca.saisonnalite(bad_ratings))
        display_fig(rrca.saisonnalite(good_ratings))
        st.write("Nous n'observons pas d'impact de la saisonnalité du post entre bad et good ratings.")

###### Page 3
    elif choice == "Influence du temps de préparation et de la complexité":
        logger.info("Naviguation - Influence du temps de préparation et de la complexité")
        #Comparaison du temps, du nombre d'étapes et du nombre d'ingrédients entre les recettes bien et mal notées
        st.subheader("Analyser l'impact du temps de préparation and la complexité sur les notes :")
        if st.session_state.df_index == 0 :
            bad_ratings, good_ratings = rrca.separate_bad_good_ratings(data, 4, 'note_moyenne')
        else :
            bad_ratings, good_ratings = rrca.separate_bad_good_ratings(data, 4, 'rating')
        data_minutes = [good_ratings['minutes'], bad_ratings['minutes']]
        display_fig(rrca.boxplot_df(data_minutes))
        data_steps = [good_ratings['n_steps'], bad_ratings['n_steps']]
        display_fig(rrca.boxplot_df(data_steps))
        data_ingred = [good_ratings['n_ingredients'], bad_ratings['n_ingredients']]
        display_fig(rrca.boxplot_df(data_ingred))
        st.write("Les recettes mal notées tendent à avoir des temps de préparation plus longs et un nombre d'étapes à suivre plus élevé. Rien à signalier sur le nombre d'ingrédients.")

        # Distribution de la note par rapport à la variable minutes / n_steps / n_ingredients en %:
        st.write("Pour aller plus loin dans l'analyse nous allons créer des bins pour chaque variable avec des seuils définis (low, medium, high) et regarder le proportion des moyennes dans chaque catégorie.")
        if st.session_state.df_index == 0 :
            fig, comparison_minutes = rrca.rating_distribution(df=data,variable='minutes',rating_var='note_moyenne',low_threshold=15,mean_range=(30, 50),high_threshold=180)
            display_fig(fig)
            st.write("Distribution de la note par rapport à la variable minutes en %:")
            st.write(comparison_minutes)
            fig, comparison_steps = rrca.rating_distribution(df=data,variable='n_steps',rating_var='note_moyenne',low_threshold=3,mean_range=(8, 10),high_threshold=15)
            display_fig(fig)
            st.write("Distribution de la note par rapport à la variable n_steps en %:")
            st.write(comparison_steps)
            fig, comparison_ingr = rrca.rating_distribution(df=data,variable='n_ingredients',rating_var='note_moyenne',low_threshold=3,mean_range=(8, 10),high_threshold=15)
            display_fig(fig)
        else :
            fig, comparison_minutes = rrca.rating_distribution(df=data,variable='minutes',rating_var='rating',low_threshold=15,mean_range=(30, 50),high_threshold=180)
            display_fig(fig)
            st.write("Distribution de la note par rapport à la variable minutes en %:")
            st.write(comparison_minutes)
            fig, comparison_steps = rrca.rating_distribution(df=data,variable='n_steps',rating_var='rating',low_threshold=3,mean_range=(8, 10),high_threshold=15)
            display_fig(fig)
            st.write("Distribution de la note par rapport à la variable n_steps en %:")
            st.write(comparison_steps)
            fig, comparison_ingr = rrca.rating_distribution(df=data,variable='n_ingredients',rating_var='rating',low_threshold=3,mean_range=(8, 10),high_threshold=15)
            display_fig(fig)

        st.write("Distribution de la note par rapport à la variable n_ingredients en %:")
        st.write(comparison_ingr)
        st.write("Même analyse pour la variable nombre d'étapes : plus les recettes ont un nombre d'étapes élevé / sont complexes plus elles sont mal notées. A contrario les recettes avec moins de 3 étapes sont sensiblement mieux notées.")
        st.write("Le nombre d'ingrédients en revanche ne semble pas impacté la moyenne.")
        
        st.write("Réalisons une régression avec ces trois variables pour comprendre dans quelle mesure elles impactent la note et si cette hypothèse est statistiquement viable.")
        st.write("La matrice de corrélation en les variables 'minutes','n_steps','n_ingredients' est la suivante")
        columns_to_analyze = ['minutes','n_steps','n_ingredients']
        correlation = rrca.correlation(data, columns_to_analyze)
        st.write(correlation)

        # Régression linéaire
        st.write("Régression linéaire entre les variables 'minutes','n_steps','n_ingredients' et la note moyenne : ")
        X = data[['minutes', 'n_steps']]
        if st.session_state.df_index == 0 :
            y = data['note_moyenne']
        else :
            y = data['rating']
        model=rrca.OLS_regression(X,y)
        st.write(model.summary())
        st.write("ANALYSE :")
        st.write("R-Squared = O.OO1 -> seulement 0.1% de la variance dans les résultats est expliquée par les variables n_steps et minutes. C'est très bas, ces variables ne semblent pas avoir de pouvoir prédictif sur les ratings, même si on a pu détecter des tendances de comportements users.")
        st.write("Prob (F-Stat) = p-value est statistiquement signifiante (car < 0.05) -> au moins un estimateur a une relation linéaire avec note_moyenne. Cependant l'effet sera minime, comme le montre le résultat R-Squared")
        st.write("Coef minute : VERY small. p-value < 0.05 donc statistiquement signifiant mais son effet est quasi négligeable sur note_moyenne. Même constat pour n_steps même si l'effet est légèrement supérieur : une augmentation de 10 étapes va baisser la moyenne d'environ 0.025...")
        st.write("Les tests Omnibus / Prob(Omnibus) et Jarque-Bera (JB) / Prob(JB) nous permettent de voir que les résidus ne suivent probablement pas une distribution gaussienne, les conditions pour une OLS ne sont donc pas remplies.")
        st.write("--> il va falloir utiliser une log transformation pour s'approcher de variables gaussiennes.")

        # 35) Régression linéaire avec log transformation
        data['minutes_log'] = rrca.dflog(data, 'minutes')
        data['n_steps_log'] = rrca.dflog(data, 'n_steps')
        X = data[['minutes_log', 'n_steps_log']]
        if st.session_state.df_index == 0 :
            y = data['note_moyenne']
        else :
            y = data['rating']
        model=rrca.OLS_regression(X,y)
        st.write(model.summary())
        st.write("ANALYSE :")
        st.write("En passant au log, on se rend compte que la variable minute a plus de poids sur la moyenne que le nombre d'étapes. Néanmoins bien que les variables minutes_log et n_steps_log soient statistiquement significatives (cf p value), leur contribution à la prédiction de la note moyenne est très faible.")
        st.write("En effet R2 est toujours extrêmement petit donc ces deux variables ont un impact minime sur la moyenne, qui ne permet pas d'expliquer les variations de la moyenne.")
        st.write("Il est probablement nécessaire d'explorer d'autres variables explicatives ou d'utiliser un modèle non linéaire pour mieux comprendre la note_moyenne.")

###### Page 4
    elif choice == "Influence du contenu nutritionnel":
        logger.info("Naviguation - Influence du contenu nutritionnel")
        st.subheader("Analyser le contenu nutritionnel des recettes et leur impact sur les notes")
        # comparaison calories
        if st.session_state.df_index == 0 :
            fig, comparison_calories = rrca.rating_distribution(df=data,variable='calories',rating_var='note_moyenne',low_threshold=100,mean_range=(250, 350),high_threshold=1000)
        else :
            fig, comparison_calories = rrca.rating_distribution(df=data,variable='calories',rating_var='rating',low_threshold=100,mean_range=(250, 350),high_threshold=1000)
        st.write("\nComparison of Rating Distribution in %:")
        display_fig(fig)
        st.write("Distribution de la note par rapport à la variable calories en %:")
        st.write(comparison_calories)
        # comparaison total_fat
        if st.session_state.df_index == 0 :
            fig, comparison_total_fat = rrca.rating_distribution(df=data,variable='total_fat',rating_var='note_moyenne',low_threshold=8,mean_range=(15, 25),high_threshold=100)
        else :
            fig, comparison_total_fat = rrca.rating_distribution(df=data,variable='total_fat',rating_var='rating',low_threshold=8,mean_range=(15, 25),high_threshold=100)
        st.write("\nComparison of Rating Distribution in %:")
        display_fig(fig)
        st.write("Distribution de la note par rapport à la variable total_fat en %:")
        st.write(comparison_total_fat)
        # comparaison sugar
        if st.session_state.df_index == 0 :
            fig, comparison_sugar = rrca.rating_distribution(df=data,variable='sugar',rating_var='note_moyenne',low_threshold=8,mean_range=(15, 25),high_threshold=60)
        else :
            fig, comparison_sugar = rrca.rating_distribution(df=data,variable='sugar',rating_var='rating',low_threshold=8,mean_range=(15, 25),high_threshold=60)
        st.write("\nComparison of Rating Distribution in %:")
        display_fig(fig)
        st.write("Distribution de la note par rapport à la variable sugar en %:")
        st.write(comparison_sugar)
        # comparaison protein
        if st.session_state.df_index == 0 :
            fig, comparison_protein = rrca.rating_distribution(df=data,variable='protein',rating_var='note_moyenne',low_threshold=8,mean_range=(15, 25),high_threshold=60)
        else :
            fig, comparison_protein = rrca.rating_distribution(df=data,variable='protein',rating_var='rating',low_threshold=8,mean_range=(15, 25),high_threshold=60)
        st.write("\nComparison of Rating Distribution in %:")
        display_fig(fig)
        st.write("Distribution de la note par rapport à la variable protein en %:")
        st.write(comparison_protein)
        # conclusion
        st.write("Les variations sont trop faibles. Les contenus nutritionnels des recettes n'impactent pas la moyenne.")

###### Page 5
    elif choice == "Influence de popularité et de la visibilité":
        logger.info("Naviguation - Influence de popularité et de la visibilité")
        #42 Analyser l'impact de la popularité des recettes sur les notes
        st.subheader("Analyser l'impact de la popularité et de la visibilité des recettes sur les notes")
        if st.session_state.df_index == 0 :
            # Calculer Q1, Q3, et IQR pour le nb_users
            Q1_nb_user = rrca.calculate_quartile(data, 'nb_user',0.25)
            Q2_nb_user = rrca.calculate_quartile(data, 'nb_user',0.50)
            Q3_nb_user = rrca.calculate_quartile(data, 'nb_user',0.75)
            st.write("Q1 pour le nombre d'utilisateurs : ", Q1_nb_user)
            st.write("Q2 pour le nombre d'utilisateurs : ", Q2_nb_user)
            st.write("Q3 pour le nombre d'utilisateurs : ", Q3_nb_user)
            # comparaison popularity
            fig, comparison_popularity = rrca.rating_distribution(df=data,variable='nb_user',rating_var='note_moyenne',low_threshold=2,mean_range=(2, 3),high_threshold=4)
            st.write("\nComparison of Rating Distribution in %:")
            display_fig(fig)
            st.write("Distribution de la note par rapport à la variable popularity en %:")
            st.write(comparison_popularity)
            # conclusion
            st.write("Il est très net ici que les recettes ayant le moins de notes sont celles les moins bien notés. Cela veut dire qu'elles sont moins populaires et/ou moins visibles. Au contraire celles avec le plus de notes sont les mieux notées.")
            st.write("Ou ça peut vouloir dire que les utilisateurs ne notent pas les mauvaises recettes. La mauvaise note appelle la mauvaise note.")
            st.write("A CREUSER :")
            st.write("- qui sont les users qui ont mal noté ces recettes : ont-ils beaucoup noté ? Mettent-ils que des mauvaises notes ? Pour vérifier si cette information est significative.")
            st.write("- faire un heatmap : nb_users/note_moyenne")
        else :
            st.write("Il ne sert à rien de passer sur le data set 2 car pour cette partie car nous traçons nb_users en fonction de la note moyenne. Merci donc de revenir sur le data set 1 pour cette analyse.")

###### Page 6
    elif choice == "Influence des tags et des descriptions":
        logger.info("Naviguation - Influence des tags et des descriptions")
        #44 Analyser des variables categorical - tags & descriptions
        st.subheader("Analyses des variables categorical - tags & descriptions - pour comprendre grâce au verbage les critères d'une mauvaise note")
        if st.session_state.df_index == 0 :
            bad_ratings, good_ratings = rrca.separate_bad_good_ratings(data, 4, 'note_moyenne')

            st.write("Analysons les tags et descriptions pour essayer de trouver des thèmes communs entre les recettes mal notées. On les comparera aux recettes bien notées. Pour cela nous utiliserons les dataframes bad_ratings et good_ratings. La première étape est de réaliser un pre-processing de ces variables (enlever les mots inutiles, tokeniser).")
            # Preprocessing des tags et descriptions
            bad_ratings.loc[:,'tags_clean'] = bad_ratings.loc[:,'tags'].fillna('').apply(rrca.preprocess_text)
            bad_ratings.loc[:,'description_clean'] = bad_ratings.loc[:,'description'].fillna('').apply(rrca.preprocess_text)
            good_ratings.loc[:,'tags_clean'] = good_ratings.loc[:,'tags'].fillna('').apply(rrca.preprocess_text)
            good_ratings.loc[:,'description_clean'] = good_ratings.loc[:,'description'].fillna('').apply(rrca.preprocess_text)
            # Mots les plus courants dans les tags des recettes mal notées
            most_common_bad_tags_clean = rrca.get_most_common_words(bad_ratings['tags_clean'])
            st.write("Les tags les plus courants dans les recettes mal notées :")
            bad_tag_words_set=rrca.extractWordFromTUpple(most_common_bad_tags_clean)
            st.write(bad_tag_words_set)
            # Mots les plus courants dans la descriptions des recettes mal notées
            most_common_bad_desciption_clean = rrca.get_most_common_words(bad_ratings['description_clean'])
            st.write("\nLes mots les plus courants dans les descriptions des recettes mal notées :")
            bad_desc_words_set=rrca.extractWordFromTUpple(most_common_bad_desciption_clean)
            st.write(bad_desc_words_set)
            # Mots les plus courants dans les tags des recettes bien notées
            most_common_good_tags_clean = rrca.get_most_common_words(good_ratings['tags_clean'])
            st.write("Les tags les plus courants dans les recettes bien notées :")
            good_tag_words_set=rrca.extractWordFromTUpple(most_common_good_tags_clean)
            st.write(good_tag_words_set)
            # Mots les plus courants dans descriptions des recettes bien notées
            most_common_good_desciption_clean = rrca.get_most_common_words(good_ratings['description_clean'])
            st.write("\nLes mots les plus courants dans les descriptions des recettes bien notées :")
            good_desc_words_set=rrca.extractWordFromTUpple(most_common_good_desciption_clean)
            st.write(good_desc_words_set)
            # Mots uniques dans les tags et descriptions des recettes mal notées :
            st.write("Mots uniques dans les tags des recettes mal notées :", rrca.uniqueTags(bad_tag_words_set, good_tag_words_set))
            st.write("Mots uniques dans les descriptions des recettes mal notées :", rrca.uniqueTags(bad_desc_words_set, good_desc_words_set)) 
        else :
            bad_ratings, good_ratings = rrca.separate_bad_good_ratings(data, 4, 'rating') # la fonction marche mais en local uniquement, trop lourde en RAM pour le serveur
            st.write("Sur le dataset 2, les calculs prennent bcp trop de temps et d'espace RAM ce qui provoquait des crashs.")
            st.write("Nous avons donc fait tourner en local les calculs et les résultats sont les suivants :")
            st.write("Mots uniques dans les tags des recettes mal notées : {'rice', 'fish'}")
            st.write("Mots uniques dans les tags des recettes mal notées : {'low', 'healthy'}")
        # Conclusion
        st.write("Il vaut mieux éviter d'écrire une recette avec les mots et les descriptions ci-dessus.")
        st.write("Notons que les résultats sont différents entre les deux datasets.")

###### Page 7
    elif choice == "Influence du temps par étape":
        logger.info("Navigation - Influence du temps par étape")
        st.write("Pour cette étude on va se concentrer sur le temps moyen par étape et non le temps total de préparation.")
        if st.session_state.df_index == 0 :
            st.write("La fonction permettant de calculer le temps moyen par étape pour chaque recette ne peut fonctionner que sur le dataset 2.")
            st.write("Merci donc de changer de dataset.")
        else :
            fig=rrca.time_per_step(data, 'minutes', 'n_steps')
            display_fig(fig)
            st.write("Plus le rapport temps par étape est élevé plus la proportion de recettes mal notée augmente.")
            # Régression linéaire
            data['minute_step'] = data['minutes'] / data['n_steps']
            X = data['minute_step']
            y = data['rating']
            model=rrca.OLS_regression(X,y)
            st.write(model.summary())
            st.write("Le modèle linéaire n'est cependant pas applicable ici. R2 est proche de 0.")

###### Page 8
    elif choice == "Analyse des profils utilisateurs" :
        logger.info("Naviguation - Analyse des profils utilisateurs")
        st.write("Pour cette étude on regarde les profils utilisateurs (contributeurs vs non-contributeurs) et leur impact sur les notes.")
        if st.session_state.df_index == 0 :
            st.write("La fonction permettant de séparer les profils utilisateurs ne peut fonctionner que sur le dataset 2.")
            st.write("Merci donc de changer de dataset.")

        else :
            user_profiles = rrca.create_dfuser_profiles(data) 
            st.dataframe(user_profiles.head())
            # distribution du nombre de raters également contributeurs
            fig=rrca.rating_isContributor(user_profiles, 'is_contributor')
            display_fig(fig)
            # Moyenne du nombre de recettes notées pour contributeurs et non-contributeurs
            contributor_stats = user_profiles.groupby('is_contributor')['num_recipes_rated'].mean()
            st.write("Moyenne du nombre de recettes notées pour contributeurs et non-contributeurs :")
            st.write(contributor_stats)
            # Distribution des notes moyennes par groupe (contributeur vs non-contributeur)
            st.write("Distribution des notes moyennes par groupe (contributeur vs non-contributeur)")
            fig=rrca.plot_distributionIsContributor(user_profiles, 'is_contributor', 'mean_rating')
            display_fig(fig)
            st.write("Les utilisateurs ne contribuant pas sont ceux qui notent le plus mal et qui sont les plus réguliers et homogènes dans leur notation.") 
            st.write("Les contributeurs sont ceux qui notent le plus de recettes et ils les notent bien. Cependant ils sont beaucoup plus dispersés dans leur notation. Ceci constitue un premier biais qui tire les notes et les moyennes vers le haut.")
            user_profiles = None # Libérer la mémoire


if __name__ == "__main__":
    main()