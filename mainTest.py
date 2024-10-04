import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns 

# Import the data and print the first three rows 
recipe = pd.read_csv("../data/RAW_recipes.csv")
print(recipe.head(3))
