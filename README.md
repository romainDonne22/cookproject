# Analyse de Recettes et Application Streamlit

## Description du Projet

Ce projet consiste en l'analyse d'une base de données de recettes contenant des informations sur les utilisateurs, les notes attribuées aux recettes, et d'autres métadonnées. L'objectif principal était de :
1. Analyser et prétraiter les données afin d'obtenir des fichiers exploitables.
2. Développer une application frontale simple et interactive avec **Streamlit** pour visualiser les résultats d'analyse.

Nous avons choisi de travailler directement avec des fichiers **CSV** au lieu d'utiliser une base de données, en prétraitant les fichiers pour en réduire la taille tout en conservant les informations pertinentes.

## Organisation du Projet

Voici une vue d'ensemble des principaux composants du projet et de leur rôle :

### **Analyse Exploratoire**
- **Notebooks Jupyter :**  
  Les premières analyses ont été effectuées dans des notebooks Jupyter partagés entre les membres de l'équipe. Ces notebooks, contenant des visualisations et des calculs exploratoires, se trouvent dans le dossier `ipynb_checkpoints`.


### **Prétraitement des Données**
- **Dossier `Pretraitement/` :**  
  Ce dossier contient les scripts de prétraitement des fichiers CSV. Les données initiales ont été nettoyées et filtrées pour générer des fichiers CSV de taille acceptable, adaptés à nos besoins et aux contraintes de Streamlit.


### **Consolidation des Analyses**
- **Fichier `rating_recipe_correlation_analysis.py` :**  
  Ce fichier regroupe toutes les analyses pertinentes réalisées dans les notebooks Jupyter. Il centralise le code nécessaire pour les calculs et visualisations liés aux notes des recettes et aux corrélations.


### **Interface Utilisateur**
- **Fichier `front.py` :**  
  Ce script génère une interface utilisateur avec **Streamlit** permettant de visualiser les résultats d'analyse. L'application est interactive et fournit une vue claire des corrélations et statistiques importantes issues de l'analyse des données.


### **Tests Unitaires**
- **Dossier `_pytest_/` :**  
  Les tests unitaires concernant les fonctions critiques du fichier `rating_recipe_correlation_analysis.py` se trouvent dans ce dossier. Ces tests assurent la robustesse et la fiabilité des résultats d'analyse.


## Fonctionnalités

1. **Nettoyage et Prétraitement des Données :**
   - Utilisation des fichiers de Kaggle RAW_recipes.csv et RAW_interactions.csv
   - Suppression des doublons et gestion des valeurs manquantes.
   - Filtrage et réduction de la taille des fichiers CSV pour optimiser les performances.

3. **Analyse des Données :**
   - Étude des corrélations entre les notes des utilisateurs et les caractéristiques des recettes.
   - Exploration des tendances dans les données de recettes.

4. **Interface Utilisateur avec Streamlit :**
   - Visualisation interactive des résultats.
   - Navigation simple et intuitive via un site généré avec Streamlit.

5. **Tests Unitaires :**
   - Vérification des fonctions principales pour assurer leur fiabilité.


## Installation et Exécution

### Prérequis
- Python 3.9 ou supérieur
- **Poetry** pour la gestion des dépendances

### Étapes d'Installation

1. Clonez le dépôt :
   ```bash
   git clone <URL_DU_DEPOT>
   cd <NOM_DU_PROJET>
   ```

2. Installez les dépendances avec Poetry :
   ```bash
   poetry install
   ```

3. Générez la documentation Sphinx (facultatif) :
   ```bash
   poetry run make -C docs html
   ```


### Lancer l'Application Streamlit

Pour démarrer l'application, exécutez :
```bash
poetry run streamlit run front.py
```
Cela ouvrira l'application dans votre navigateur par défaut.


### Lancer les Tests Unitaires

Pour exécuter les tests unitaires :
```bash
poetry run pytest _pytest_/
```


## Structure du Projet

```
project-root/
├── docs/                             # Documentation générée avec Sphinx
├── Pretraitement/                    # Scripts de prétraitement des fichiers CSV
├── ipynb_checkpoints/                # Notebooks Jupyter pour analyse initiale
├── _pytest_/                         # Tests unitaires
├── application.py                    # Back-end (projet spécifique)
├── front.py                          # Interface Streamlit
├── rating_recipe_correlation_analysis.py  # Analyses consolidées
├── README.md                         # Documentation du projet
├── pyproject.toml                    # Configuration Poetry
├── poetry.lock                       # Verrouillage des dépendances Poetry
```

## Auteur

Réalisé par Aude, Camille, Romain.







