import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns 

def datasetAnalysis(data):
    # Print the first three rows of the data
    print(data.head(3))

    # Print the shape of the data
    num_ligne = data.shape[0]
    print("Le nombre de lignes dans le dataset est: ", num_ligne)
    num_columns = data.shape[1]
    print("Le nombre de colonnes dans le dataset est : ", num_columns)

    # Print the missing values
    print("Les missing values sont : ", data.isnull().sum().sum())
    print("Par varaible",data.isna().sum())

# Import the data and print the first three rows 
#recipe = pd.read_csv("../data/RAW_recipes.csv")
recipe = pd.read_csv("../data/PP_users.csv")
# Analyse the dataset RAW_recipes
datasetAnalysis(recipe)