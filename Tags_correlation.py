import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import ast

# Charger les données depuis le fichier CSV
df = pd.read_csv('RAW_recipes.csv')

# Définir une bonne note (par exemple, >= 4)
good_rating_threshold = 4

# Charger les interactions pour obtenir les notes
interactions = pd.read_csv('RAW_interactions.csv')

# Filtrer les interactions avec de bonnes notes
good_ratings = interactions[interactions['rating'] >= good_rating_threshold]

# Sélectionne les recettes dont l'ID est présent dans les interactions filtrées (celles avec de bonnes notes).
good_recipes = df[df['id'].isin(good_ratings['recipe_id'])]

# Extraire les tags et compter leur fréquence
tags_good = good_recipes['tags'].apply(ast.literal_eval).explode()
tag_counts_good = Counter(tags_good)

# Convertir en DataFrame pour la visualisation
tag_counts_df_good = pd.DataFrame(tag_counts_good.items(), columns=['tag', 'count']).sort_values(by='count', ascending=False)

# Définir une mauvaise note (par exemple, <= 2)
bad_rating_threshold = 2

# Filtrer les interactions avec de mauvaises notes
bad_ratings = interactions[interactions['rating'] <= bad_rating_threshold]

# Sélectionne les recettes dont l'ID est présent dans les interactions filtrées (celles avec de mauvaises notes).
bad_recipes = df[df['id'].isin(bad_ratings['recipe_id'])]

# Extraire les tags et compter leur fréquence
tags_bad = bad_recipes['tags'].apply(ast.literal_eval).explode()
tag_counts_bad = Counter(tags_bad)

# Convertir en DataFrame pour la visualisation
tag_counts_df_bad = pd.DataFrame(tag_counts_bad.items(), columns=['tag', 'count']).sort_values(by='count', ascending=False)

# Visualiser les tendances des tags
plt.figure(figsize=(24, 8))

# Sous-graphe pour les bonnes notes
plt.subplot(1, 2, 1)
plt.barh(tag_counts_df_good['tag'].head(20), tag_counts_df_good['count'].head(20))
plt.xlabel('Fréquence')
plt.ylabel('Tags')
plt.title('Tendances des tags pour les recettes avec de bonnes notes')
plt.gca().invert_yaxis()

# Sous-graphe pour les mauvaises notes
plt.subplot(1, 2, 2)
plt.barh(tag_counts_df_bad['tag'].head(20), tag_counts_df_bad['count'].head(20))
plt.xlabel('Fréquence')
plt.ylabel('Tags')
plt.title('Tendances des tags pour les recettes avec de mauvaises notes')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

# import pandas as pd
# import ast
# from sklearn.preprocessing import MultiLabelBinarizer

# # Charger les données depuis le fichier CSV
# df = pd.read_csv('RAW_recipes.csv')
# interactions = pd.read_csv('RAW_interactions.csv')

# # Calculer la note moyenne pour chaque recette
# average_ratings = interactions.groupby('recipe_id')['rating'].mean().reset_index()
# average_ratings.columns = ['id', 'average_rating']

# # Joindre les notes moyennes au DataFrame des recettes
# df = df.merge(average_ratings, on='id')

# # Extraire les tags et créer une matrice binaire
# df['tags'] = df['tags'].apply(ast.literal_eval)
# mlb = MultiLabelBinarizer()
# tags_matrix = mlb.fit_transform(df['tags'])
# tags_df = pd.DataFrame(tags_matrix, columns=mlb.classes_)

# # Ajouter les notes moyennes à la matrice des tags
# tags_df['average_rating'] = df['average_rating']

# # Calculer la corrélation
# correlation_matrix = tags_df.corr()

# # Afficher la matrice de corrélation des tags avec les notes moyennes
# print(correlation_matrix['average_rating'].sort_values(ascending=False))