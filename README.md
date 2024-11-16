Test

Les fichiers d'interactions représentent tout le dataset diviser en trois csv

Colonnes de interactions_test.csv /Colonnes de interactions_validation.csv /Colonnes de interactions_train.csv:

['user_id', 'recipe_id', 'date', 'rating', 'u', 'i']

Colonnes de PP_recipes.csv:

['id', 'i', 'name_tokens', 'ingredient_tokens', 'steps_tokens','techniques', 'calorie_level', 'ingredient_ids']

Colonnes de PP_users.csv:

['u', 'techniques', 'items', 'n_items', 'ratings', 'n_ratings']

Colonnes de RAW_interactions.csv:

['user_id', 'recipe_id', 'date', 'rating', 'review']

Colonnes de RAW_recipes.csv:

['name', 'id', 'minutes', 'contributor_id', 'submitted', 'tags', 'nutrition', 'n_steps', 'steps', 'description', 'ingredients','n_ingredients']

ATTENTION : u et i represente les identifiant utilisateur mappé à des entiers contigus à partir de 0.
u et i sont les ids de user_id et recipe_id reformulé pour simplifié les algos de ML 
On n'utilisera pas le fichier pckl qui contient sans doute l'algo de ML 
Initialisation de l'environnement Poetry avec les dépendances suivantes : 

Environement Poetry : 

python = "^3.10.0"
matplotlib = "^3.9.2"
numpy = "^2.1.1"
streamlit = "^1.39.0"
pandas = "^2.2.3"
seaborn = "^0.13.2"
pytest = "^8.3.3"
Sphinx = "^8.0.2"
pycodestyle = "^2.12.1"

commande pour lancer l'application
poetry update
poetry shell
run appli pour le moment : python mainTest.py  / python3 mainTest.py
run appli front : streamlit ru front.py

Process Git :

git branch : savoir dans quelle branche je me situe
git checkout -b <nombranche> : créer une nouvelle branche et switch dessus

git add . : ajouter toutes les modifications que j'ai réalisé sur ma branche
OU
git add fichier.py : ajouter les modif du fichier.py uniquement

git commit -m "Message pour expliquer à quoi correspond mon commit" : très important pour suivre l'historique des commit dans GitHub

git push origin <branch> 
Une fois les modif push, aller sur GitHub pour créer une "Pull Request" qui permettra au groupe de vérifier les modifications des autres et commenter si besoin.
On peut faire plusieurs commits dans une même Pull Request

Pour récupérer les modif dans main :
git checkout main : se replacer dans main
git pull origin <branch> : on récupère les modifs de la branche spécifique dans main

A la fin du travail sur la branche, cliquer sur "Merge Pull Request" dans Git -> merge la branche et main
git branch -d nom_branche : supprimer la branche sur laquelle on a fini de travailler

git status : où en sont les push et pull sur les différentes branches, pour éviter des conflits entre push/pull
git log : check les commits







