Titre

Initialisation de l'environnement Poetry avec les dépendances suivantes : 

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

Best practices Git :

git branch : savoir dans quelle branche je me situe
git checkout -b <nombranche> : créer une nouvelle branche et switch dessus

git add . : ajouter toutes les modifications que j'ai réalisé sur ma branche
git add fichier.py : ajouter les modif du fichier.py uniquement

git commit -m "Message pour expliquer à quoi correspond mon commit" : très important pour suivre l'historique des commit dans GitHub

git push origin <branch> 
Une fois les modif push, aller sur GitHub pour créer une "Pull Request" qui permettra au groupe de vérifier les modifications des autres et commenter si besoin.





