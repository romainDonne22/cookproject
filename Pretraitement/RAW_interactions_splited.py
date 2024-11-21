import pandas as pd

# Charger le DataFrame
fichierCSV = pd.read_csv("../Data/RAW_interactions.csv")
# Taille maximale du fichier en octets (100 Mo)
max_file_size = 100 * 1024 * 1024
# Estimer la taille de chaque ligne
row_size = fichierCSV.memory_usage(index=True, deep=True).sum() / len(fichierCSV)
# Calculer le nombre de lignes par fichier
rows_per_file = int(max_file_size / row_size)
# Diviser le DataFrame en morceaux
num_files = (len(fichierCSV) + rows_per_file - 1) // rows_per_file  # Calculer le nombre de fichiers nécessaires
for i in range(num_files):
    start_row = i * rows_per_file
    end_row = min((i + 1) * rows_per_file, len(fichierCSV))
    chunk = fichierCSV.iloc[start_row:end_row]
    chunk.to_csv(f"RAW_interactions_part_{i + 1}.csv", index=False)
    print(f"RAW_interactions_part_{i + 1}.csv créé avec succès")