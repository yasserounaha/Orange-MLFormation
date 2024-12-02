

longueur_sépale = 5.1
print("La longueur du sépale est :", longueur_sépale)
print("------------------------------------------------")
longueurs_sépales = [5.1, 4.9, 4.7, 4.6, 5.0]
print("Longueurs des sépales :", longueurs_sépales)
print("------------------------------------------------")

import pandas as pd
df = pd.DataFrame(longueurs_sépales, columns=["Longueur_Sépale"])
print(df)
print("------------------------------------------------")
longueur_sépale = 5.1
if longueur_sépale > 5:
    print("Le sépale est long.")
else:
    print("Le sépale est court.")
print("------------------------------------------------")
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df[df["sepal length (cm)"] > 5])
print("------------------------------------------------")

import matplotlib.pyplot as plt
plt.scatter(df["sepal length (cm)"], df["sepal width (cm)"])
plt.xlabel("Longueur des sépales")
plt.ylabel("Largeur des sépales")
plt.title("Relation entre longueur et largeur des sépales")
plt.show()
print("------------------------------------------------")


plt.scatter(df["sepal length (cm)"], df["sepal width (cm)"], c=iris.target, cmap='viridis')
plt.xlabel("Longueur des sépales")
plt.ylabel("Largeur des sépales")
plt.title("Séparation des espèces dans Iris")
plt.show()
print("------------------------------------------------")
def calculer_moyenne(liste):
    return sum(liste) / len(liste)
print("Moyenne des longueurs :", calculer_moyenne(longueurs_sépales))
print("------------------------------------------------")
def standardiser(valeurs):
 moyenne = sum(valeurs) / len(valeurs)
 ecart_type = (sum([(x - moyenne)**2 for x in valeurs]) / len(valeurs))**0.5
 return [(x - moyenne) / ecart_type for x in valeurs]
print("Longueurs standardisées :", standardiser(longueurs_sépales))
print("------------------------------------------------")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
print("Précision :", model.score(X_test, y_test))
