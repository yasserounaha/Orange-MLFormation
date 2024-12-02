import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
# Chargement des données
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
# Visualisation rapide
df.plot(kind='scatter', x='sepal length (cm)', y='sepal width (cm)')
plt.title("Exemple de visualisation avec Matplotlib")
plt.show()
