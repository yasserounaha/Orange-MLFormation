import pandas as pd

# Création du DataFrame
data = {
    'Nom': ['Alice', 'Bob', 'Charlie', 'David'],
    'Âge': [25, 30, 35, 28],
    'Salaire': [50000, 60000, 75000, 55000],
    'Département': ['RH', 'Tech', 'Finance', 'Marketing']
}
df = pd.DataFrame(data)

# 1. Afficher les 3 premières lignes
print("Les 3 premières lignes:")
print(df.head(3))
print("\n")

# 2. Calculer l'âge moyen
age_moyen = df['Âge'].mean()
print(f"L'âge moyen des employés est: {age_moyen:.2f} ans")
print("\n")

# 3. Filtrer les employés qui gagnent plus de 55000
salaires_eleves = df[df['Salaire'] > 55000]
print("Employés gagnant plus de 55000:")
print(salaires_eleves)


print("------------------------------------------------")

import pandas as pd
import numpy as np

# Création d'un DataFrame plus complet pour une meilleure analyse
data = {
    'Nom': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Henry'],
    'Âge': [25, 30, 35, 28, 45, 33, 29, 41],
    'Salaire': [50000, 60000, 75000, 55000, 80000, 65000, 52000, 70000],
    'Département': ['RH', 'Tech', 'Finance', 'Marketing', 'Finance', 'Tech', 'Marketing', 'RH'],
    'Expérience': [2, 5, 8, 3, 15, 7, 4, 12]
}
df = pd.DataFrame(data)

# 1. Statistiques descriptives complètes
print("Statistiques descriptives complètes:")
print(df.describe())
print("\n")

# 2. Analyse par département
print("Analyse par département:")
dept_analysis = df.groupby('Département').agg({
    'Salaire': ['mean', 'min', 'max', 'count'],
    'Âge': 'mean',
    'Expérience': 'mean'
})
print(dept_analysis)
print("\n")

# 3. Calcul des corrélations
print("Matrice de corrélation:")
correlation_matrix = df[['Âge', 'Salaire', 'Expérience']].corr()
print(correlation_matrix)
print("\n")

# 4. Statistiques personnalisées
print("Statistiques personnalisées:")
custom_stats = {
    'Salaire moyen': df['Salaire'].mean(),
    'Salaire médian': df['Salaire'].median(),
    'Écart-type salaire': df['Salaire'].std(),
    'Âge moyen par département': df.groupby('Département')['Âge'].mean(),
    'Nombre employés par département': df['Département'].value_counts()
}

for stat_name, value in custom_stats.items():
    print(f"\n{stat_name}:")
    print(value)

# 5. Analyse des quartiles pour le salaire
print("\nAnalyse des quartiles pour le salaire:")
quartiles = df['Salaire'].quantile([0.25, 0.5, 0.75])
print(quartiles)

# 6. Visualisation simple avec Pandas
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

# Graphique 1: Salaire moyen par département
plt.subplot(131)
df.groupby('Département')['Salaire'].mean().plot(kind='bar')
plt.title('Salaire moyen par département')
plt.xticks(rotation=45)

# Graphique 2: Distribution des âges
plt.subplot(132)
df['Âge'].hist()
plt.title('Distribution des âges')

# Graphique 3: Relation Expérience-Salaire
plt.subplot(133)
plt.scatter(df['Expérience'], df['Salaire'])
plt.xlabel('Expérience')
plt.ylabel('Salaire')
plt.title('Expérience vs Salaire')

plt.tight_layout()
plt.show()

print("------------------------------------------------")


import pandas as pd

# Supposons que vous ayez un fichier 'donnees_ventes.csv'
# Exemple de données fictives pour l'illustration
data = {
    'Produit': ['A', 'B', 'C', 'D', 'E'],
    'Prix_Vente': [200, 150, 300, 250, None],
    'Coût_Production': [120, 100, 200, 180, 150],
    'Quantité_Vendue': [30, 50, 20, 40, 10]
}
df_ventes = pd.DataFrame(data)

# 1. Gestion des valeurs manquantes
# Suppression des lignes avec des valeurs manquantes
df_ventes.dropna(inplace=True)

# 2. Création d'une nouvelle colonne calculée
# Calcul de la marge
df_ventes['Marge'] = df_ventes['Prix_Vente'] - df_ventes['Coût_Production']

# 3. Tri du DataFrame par marge décroissante
df_ventes_tries = df_ventes.sort_values('Marge', ascending=False)

# Affichage du DataFrame trié
print("DataFrame trié par marge décroissante:")
print(df_ventes_tries)

print("------------------------------------------------")
import pandas as pd
import matplotlib.pyplot as plt

# Exemple de données
data = {
    'Nom': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Henry'],
    'Âge': [25, 30, 35, 28, 45, 33, 29, 41],
    'Salaire': [50000, 60000, 75000, 55000, 80000, 65000, 52000, 70000],
    'Département': ['RH', 'Tech', 'Finance', 'Marketing', 'Finance', 'Tech', 'Marketing', 'RH']
}
df = pd.DataFrame(data)

# 1. Histogramme des salaires
plt.figure(figsize=(10, 6))
df['Salaire'].hist(bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution des Salaires')
plt.xlabel('Salaire')
plt.ylabel('Fréquence')
plt.grid(False)
plt.show()

# 2. Graphique à barres des salaires moyens par département
df.groupby('Département')['Salaire'].mean().plot(kind='bar', color='coral')
plt.title('Salaire Moyen par Département')
plt.ylabel('Salaire Moyen')
plt.xticks(rotation=45)
plt.show()


# 1. Filtrage avec plusieurs conditions
employes_selectionnes = df[
    (df['Âge'] > 25) & 
    (df['Salaire'] > 55000) & 
    (df['Département'] != 'RH')
]

print("Employés sélectionnés:")
print(employes_selectionnes)

# 2. Sélection de colonnes spécifiques
infos_essentielles = df[['Nom', 'Département', 'Salaire']]

print("\nInformations essentielles:")
print(infos_essentielles)

# 3. Manipulation de colonnes
# Ajout d'une colonne 'Salaire_Annuel' en multipliant le salaire mensuel par 12
df['Salaire_Annuel'] = df['Salaire'] * 12

print("\nDataFrame avec Salaire Annuel:")
print(df)