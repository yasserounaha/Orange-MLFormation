

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
