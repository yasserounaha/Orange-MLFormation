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
print("------------------------------------------------")

import numpy as np
import matplotlib.pyplot as plt

# Mêmes données et calculs que précédemment
# ...

# Création d'un graphique plus détaillé
plt.figure(figsize=(12, 8))

# Tracer les points et la ligne de régression
plt.scatter(X, Y, color='blue', label='Points de données', s=100)
plt.plot(X, Y_pred, color='red', label=f'y = {m:.2f}x + {b:.2f}', linewidth=2)

# Ajouter les lignes de résidus
for i in range(len(X)):
    plt.vlines(X[i], Y[i], Y_pred[i], colors='green', linestyles='dashed', alpha=0.5)

# Personnalisation avancée
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.title('Régression Linéaire avec Résidus', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# Ajouter des annotations pour les résidus
for i in range(len(X)):
    residual = Y[i] - Y_pred[i]
    plt.annotate(f'Résidu: {residual:.2f}', 
                xy=(X[i], Y[i]),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=8)

plt.show()

# Calcul et affichage de l'erreur moyenne quadratique (MSE)
mse = np.mean((Y - Y_pred)**2)
print(f"Erreur moyenne quadratique (MSE) : {mse:.2f}")
print("------------------------------------------------")

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def predict_with_confidence(x, X, Y, m, b, confidence=0.95):
    """
    Prédit Y avec intervalle de confiance
    """
    # Prédiction
    y_pred = m * x + b
    
    # Calcul de l'erreur standard
    n = len(X)
    y_mean = np.mean(Y)
    se = np.sqrt(np.sum((Y - (m * X + b))**2) / (n-2))
    
    # Intervalle de confiance
    x_mean = np.mean(X)
    x_std = np.std(X)
    
    pi = t * se * np.sqrt(1 + 1/n + (x - x_mean)**2 / (n * x_std**2))
    
    return y_pred, y_pred - pi, y_pred + pi

# Utilisation de la fonction avec intervalle de confiance
t = stats.t.ppf(0.975, len(X)-2)  # Pour 95% de confiance
predictions = []
lower_bounds = []
upper_bounds = []

for x in new_X:
    pred, lower, upper = predict_with_confidence(x, X, Y, m, b)
    predictions.append(pred)
    lower_bounds.append(lower)
    upper_bounds.append(upper)

# Affichage avec intervalles de confiance
plt.figure(figsize=(12, 6))
plt.scatter(X, Y, color='blue', label='Données d'entraînement')
plt.plot(X_line, Y_line, color='red', label='Régression')
plt.scatter(new_X, predictions, color='green', label='Prédictions')

# Ajout des intervalles de confiance
plt.fill_between(new_X, lower_bounds, upper_bounds, color='gray', alpha=0.2, 
                label='Intervalle de confiance 95%')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Régression Linéaire avec Intervalles de Confiance')
plt.legend()
plt.grid(True)
plt.show()

# Affichage des prédictions avec intervalles
print("\nPrédictions avec intervalles de confiance (95%) :")
for x, pred, lower, upper in zip(new_X, predictions, lower_bounds, upper_bounds):
    print(f"X = {x}:")
    print(f"  Prédiction: {pred:.2f}")
    print(f"  Intervalle: [{lower:.2f}, {upper:.2f}]")