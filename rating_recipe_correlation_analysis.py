import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import statsmodels.api as sm
#import re
#from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
#from collections import Counter
#from textblob import TextBlob
#from wordcloud import WordCloud
#from sklearn.preprocessing import StandardScaler

def load_data(fichier1, fichier2):
    try:
        data1 = pd.read_csv(fichier1)
        data2 = pd.read_csv(fichier2)
        merged_data = pd.merge(data2, data1, left_on="id", right_on="recipe_id", how="left")
        return merged_data
    except Exception as e:
        print(f"Failed to load data: {e}")
        return pd.DataFrame()

def analysisData(merged_data):
        # Dropons la colonne id en double et la colonne nutrition déjà traitée
        merged_data.drop(['recipe_id','nutrition','steps'], axis=1, inplace=True)
        merged_data.columns = ['name', 'recipe_id', 'minutes', 'contributor_id', 'submitted', 'tags',
       'n_steps', 'description', 'ingredients',
       'n_ingredients', 'calories', 'total_fat', 'sugar', 'sodium',
       'protein', 'saturated_fat', 'carbohydrates', 'year',
       'month', 'day', 'day_of_week', 'nb_user', 'note_moyenne',
       'note_mediane', 'note_q1', 'note_q2', 'note_q3', 'note_q4', 'note_max',
       'note_min', 'nb_note_lt_5', 'nb_note_eq_5']
        merged_data['recipe_id'] = merged_data['recipe_id'].astype('object')
        merged_data['contributor_id'] = merged_data['recipe_id'].astype('object')
        merged_data['year'] = merged_data['year'].astype('object')
        merged_data['month'] = merged_data['month'].astype('object')
        merged_data['day'] = merged_data['day'].astype('object')

        # Liste pour stocker les figures
        figures = []

        fig1 = plt.figure(figsize=(12, 5))
        # Distribution de la moyenne
        plt.subplot(1, 2, 1)
        plt.hist(merged_data['note_moyenne'], bins=20)
        plt.title('Distribution de la moyenne')
        plt.xlabel('Moyenne')
        plt.ylabel('Fréquence')
        # Distribution de la médiane
        plt.subplot(1, 2, 2)
        plt.hist(merged_data['note_mediane'], bins=20)
        plt.title('Distribution de la médiane')
        plt.xlabel('Médiane')
        plt.ylabel('Fréquence')
        plt.tight_layout()
        figures.append(fig1)

        # Commençons par regarder les corrélations grâce à une matrice de corrélation
        fig2 = plt.figure(figsize=(12, 8))
        correlation = merged_data[['note_moyenne', 'minutes', 'n_steps', 'n_ingredients', 'calories', 'total_fat', 'sugar', 'sodium','protein', 'saturated_fat', 'carbohydrates', 'nb_user']].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm')
        plt.title('Matrice de correlation des variables avec la moyenne')
        figures.append(fig2)
        
        return figures
    


    
