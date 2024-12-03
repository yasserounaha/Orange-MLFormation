import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    """
    Calcule la fonction logistique (sigmoid)
    σ(z) = 1 / (1 + e^(-z))
    """
    return 1 / (1 + np.exp(-z))

# Création des données
z = np.linspace(-10, 10, 100)
sigma = sigmoid(z)

# Création du graphique
plt.figure(figsize=(12, 6))

# Tracer la courbe sigmoid
plt.plot(z, sigma, 'b-', linewidth=2, label='Fonction sigmoid')

# Ajouter des lignes horizontales pour y=0 et y=1
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axhline(y=1, color='k', linestyle='-', alpha=0.3)
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)

# Ajouter une ligne verticale pour x=0
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

# Points importants
plt.plot(0, 0.5, 'ro', label='Point central (0, 0.5)')

# Personnalisation du graphique
plt.title("Fonction Logistique (Sigmoid)", fontsize=14)
plt.xlabel("z", fontsize=12)
plt.ylabel("σ(z)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()

# Ajouter des annotations
plt.annotate('Asymptote y=1', xy=(8, 0.95), xytext=(8, 0.8),
            arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate('Asymptote y=0', xy=(-8, 0.05), xytext=(-8, 0.2),
            arrowprops=dict(facecolor='black', shrink=0.05))

plt.ylim(-0.1, 1.1)
plt.show()

# Démonstration des propriétés importantes
print("Propriétés importantes de la fonction sigmoid :")
print(f"σ(-∞) ≈ {sigmoid(-100):.4f}")
print(f"σ(0) = {sigmoid(0):.4f}")
print(f"σ(∞) ≈ {sigmoid(100):.4f}")

# Exemple de valeurs
test_values = [-5, -2, 0, 2, 5]
print("\nExemples de valeurs :")
for z in test_values:
    print(f"σ({z:2d}) = {sigmoid(z):.4f}")

print("------------------------------------------------")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# 1. Génération des données
X, y = make_classification(n_samples=200, 
                         n_features=2, 
                         n_classes=2, 
                         n_clusters_per_class=1,
                         n_redundant=0,
                         random_state=42)

# 2. Préparation des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Standardisation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Création et entraînement du modèle
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Prédictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

# 6. Évaluation du modèle
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# 7. Visualisation des résultats
plt.figure(figsize=(15, 5))

# 7.1 Données et frontière de décision
plt.subplot(121)
def plot_decision_boundary(X, y, model, scaler):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))
    
    X_plot = np.c_[xx.ravel(), yy.ravel()]
    X_plot_scaled = scaler.transform(X_plot)
    Z = model.predict(X_plot_scaled)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)

plot_decision_boundary(X, y, model, scaler)
plt.title("Frontière de décision")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# 7.2 Probabilités de prédiction
plt.subplot(122)
plt.scatter(y_pred_proba[:, 0], y_pred_proba[:, 1], c=y_test)
plt.plot([0, 1], [1, 0], 'r--')
plt.xlabel("Probabilité classe 0")
plt.ylabel("Probabilité classe 1")
plt.title("Probabilités de prédiction")

plt.tight_layout()
plt.show()

# 8. Affichage des métriques
print("\nRésultats de la classification :")
print(f"Précision du modèle : {accuracy:.2f}")
print("\nMatrice de confusion :")
print(conf_matrix)
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# 9. Analyse des coefficients
print("\nCoefficients du modèle :")
for i, coef in enumerate(model.coef_[0]):
    print(f"Feature {i+1}: {coef:.4f}")
print(f"Intercept: {model.intercept_[0]:.4f}")


print("------------------------------------------------")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def create_datasets():
    """Crée trois jeux de données différents"""
    # Données linéairement séparables
    X1, y1 = make_classification(n_samples=100, n_features=2, n_redundant=0,
                               n_clusters_per_class=1, random_state=42)
    
    # Données en forme de lune (non linéairement séparables)
    X2, y2 = make_moons(n_samples=100, noise=0.15, random_state=42)
    
    # Données en cercles concentriques
    X3, y3 = make_circles(n_samples=100, noise=0.15, random_state=42)
    
    return [(X1, y1, "Données linéairement séparables"),
            (X2, y2, "Données en forme de lune"),
            (X3, y3, "Données en cercles")]

def plot_decision_boundary(model, X, y, title):
    """Trace la frontière de décision et les points"""
    # Création de la grille
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    # Prédiction pour chaque point de la grille
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Tracer la frontière de décision
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    plt.contour(xx, yy, Z, colors='k', linewidths=0.5)
    
    # Tracer les points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edcolors='k')
    
    # Personnalisation du graphique
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(scatter)

# Création des sous-graphiques
plt.figure(figsize=(15, 5))
datasets = create_datasets()

for i, (X, y, title) in enumerate(datasets, 1):
    # Standardisation des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Entraînement du modèle
    model = LogisticRegression(random_state=42)
    model.fit(X_scaled, y)
    
    # Score du modèle
    score = model.score(X_scaled, y)
    
    # Création du sous-graphique
    plt.subplot(1, 3, i)
    plot_decision_boundary(model, X_scaled, y, f"{title}\nPrécision: {score:.2f}")

plt.tight_layout()
plt.show()

# Version améliorée avec comparaison de différents classifieurs
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def compare_classifiers(X, y, title):
    """Compare différents classifieurs sur un même jeu de données"""
    classifiers = [
        ('Régression Logistique', LogisticRegression(random_state=42)),
        ('SVM', SVC(kernel='rbf', random_state=42)),
        ('Réseau de neurones', MLPClassifier(hidden_layer_sizes=(10,), random_state=42))
    ]
    
    plt.figure(figsize=(15, 5))
    
    for i, (name, clf) in enumerate(classifiers, 1):
        # Standardisation
        X_scaled = StandardScaler().fit_transform(X)
        
        # Entraînement
        clf.fit(X_scaled, y)
        score = clf.score(X_scaled, y)
        
        # Visualisation
        plt.subplot(1, 3, i)
        plot_decision_boundary(clf, X_scaled, y, f"{name}\nPrécision: {score:.2f}")
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Comparaison des classifieurs sur chaque jeu de données
for X, y, title in datasets:
    compare_classifiers(X, y, title)