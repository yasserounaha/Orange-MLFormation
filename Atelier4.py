import numpy as np
import matplotlib.pyplot as plt

# 1. Définition des données
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 5, 4, 5])

# 2. Calcul des moyennes
mean_X = np.mean(X)  # moyenne de X
mean_Y = np.mean(Y)  # moyenne de Y

# 3. Calcul de la pente (m)
# Covariance(X,Y) / Variance(X)
numerator = np.sum((X - mean_X) * (Y - mean_Y))    # Calcul du numérateur (covariance)
denominator = np.sum((X - mean_X)**2)              # Calcul du dénominateur (variance)
m = numerator / denominator

# 4. Calcul de l'ordonnée à l'origine (b)
b = mean_Y - m * mean_X

# 5. Affichage des résultats
print(f"Pente (m) : {m:.2f}")
print(f"Ordonnée à l'origine (b) : {b:.2f}")
print(f"L'équation de la ligne : y = {m:.2f}x + {b:.2f}")

# Création des points pour la ligne de régression
X_line = np.linspace(min(X), max(X), 100)
Y_line = m * X_line + b

# Tracé
plt.scatter(X, Y, color='blue', label='Points de données')
plt.plot(X_line, Y_line, color='red', label='Ligne de régression')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()