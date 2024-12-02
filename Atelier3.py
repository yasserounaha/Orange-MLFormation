import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import numpy as np

# 1. Chargement des données
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# 2. Analyse descriptive basique
print("Aperçu des données:")
print(df.head())
print("\nInformations sur le dataset:")
print(df.info())
print("\nStatistiques descriptives:")
print(df.describe())

# 3. Visualisations

# Configuration de la taille des graphiques
plt.style.use('seaborn')
fig = plt.figure(figsize=(15, 10))

# 3.1 Distribution des caractéristiques
plt.subplot(2, 2, 1)
df.boxplot()
plt.title('Distribution des caractéristiques')
plt.xticks(rotation=45)

# 3.2 Histogrammes
plt.subplot(2, 2, 2)
for feature in iris.feature_names:
    sns.kdeplot(data=df[feature], label=feature)
plt.title('Densité des distributions')
plt.legend()

# 3.3 Pairplot pour voir les relations entre variables
sns.pairplot(df, hue='species', diag_kind='kde')
plt.suptitle('Pairplot des caractéristiques par espèce', y=1.02)

# 3.4 Matrice de corrélation
plt.figure(figsize=(8, 6))
correlation_matrix = df.drop('species', axis=1).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matrice de corrélation')

# 4. Statistiques par espèce
print("\nStatistiques par espèce:")
print(df.groupby('species').describe())

# 5. Analyse des valeurs aberrantes
def detect_outliers(df, features):
    outliers_dict = {}
    for feature in features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[feature] < (Q1 - 1.5 * IQR)) | 
                     (df[feature] > (Q3 + 1.5 * IQR))]
        outliers_dict[feature] = len(outliers)
    return outliers_dict

outliers = detect_outliers(df, iris.feature_names)
print("\nNombre de valeurs aberrantes par caractéristique:")
print(outliers)

# 6. Visualisation des moyennes par espèce
plt.figure(figsize=(10, 6))
df.groupby('species').mean().plot(kind='bar')
plt.title('Moyennes des caractéristiques par espèce')
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

print("------------------------------------------------")

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Charger le dataset Iris
iris = load_iris()

# Séparer les features (X) et le target (y)
X = iris.data
y = iris.target

# Diviser le dataset en jeux d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Afficher les tailles des jeux d'entraînement et de test
print(f"Taille du jeu d'entraînement : {X_train.shape}")
print(f"Taille du jeu de test : {X_test.shape}")
print("------------------------------------------------")
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Chargement et préparation des données
iris = load_iris()
X = iris.data
y = iris.target

# Division en sets d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Création et entraînement du modèle
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 3. Prédictions
y_pred = rf_model.predict(X_test)

# 4. Évaluation du modèle
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nRapport de classification :")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 5. Visualisation de la matrice de confusion
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title('Matrice de confusion')
plt.xlabel('Prédictions')
plt.ylabel('Vraies valeurs')
plt.show()

# 6. Importance des caractéristiques
feature_importance = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Importance des caractéristiques')
plt.show()

# 7. Visualisation des prédictions vs réalité
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(range(len(y_test)), y_test, c='blue', label='Vraies valeurs')
plt.scatter(range(len(y_pred)), y_pred, c='red', label='Prédictions')
plt.title('Prédictions vs Vraies valeurs')
plt.legend()

# 8. Courbes d'apprentissage
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    rf_model, X, y, cv=5, n_jobs=-1, 
    train_sizes=np.linspace(0.1, 1.0, 10))

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), label='Score d\'entraînement')
plt.plot(train_sizes, test_scores.mean(axis=1), label='Score de validation')
plt.xlabel('Taille du jeu d\'entraînement')
plt.ylabel('Score')
plt.title('Courbes d\'apprentissage')
plt.legend()
plt.show()

# 9. Analyse détaillée des erreurs
errors = y_test != y_pred
if np.any(errors):
    print("\nAnalyse des erreurs de classification:")
    print("Index des erreurs:", np.where(errors)[0])
    print("\nDétails des erreurs:")
    for idx in np.where(errors)[0]:
        print(f"Index {idx}:")
        print(f"Vraie valeur: {iris.target_names[y_test[idx]]}")
        print(f"Prédiction: {iris.target_names[y_pred[idx]]}")
        print(f"Caractéristiques: {X_test[idx]}\n")

# 10. Probabilités de prédiction
probas = rf_model.predict_proba(X_test)
print("\nProbabilités de prédiction pour les 5 premiers échantillons:")
for i in range(5):
    print(f"\nÉchantillon {i+1}:")
    for j, species in enumerate(iris.target_names):
        print(f"{species}: {probas[i][j]:.3f}")
print("------------------------------------------------")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# 1. Chargement et préparation des données
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = boston.target

# 2. Normalisation des features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Division train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 4. Création et entraînement du modèle
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Prédictions
y_pred = model.predict(X_test)

# 6. Évaluation du modèle
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Métriques d'évaluation:")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")

# 7. Visualisations

# 7.1 Prédictions vs Réalité
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Prix réel')
plt.ylabel('Prix prédit')
plt.title('Prédictions vs Réalité')
plt.tight_layout()
plt.show()

# 7.2 Distribution des erreurs
errors = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(errors, kde=True)
plt.title('Distribution des erreurs de prédiction')
plt.xlabel('Erreur')
plt.ylabel('Fréquence')
plt.show()

# 7.3 Importance des features
feature_importance = pd.DataFrame({
    'Feature': boston.feature_names,
    'Coefficient': model.coef_
})
feature_importance['Abs_Coefficient'] = abs(feature_importance['Coefficient'])
feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='Abs_Coefficient', y='Feature', data=feature_importance)
plt.title('Importance des caractéristiques')
plt.show()

# 8. Analyse détaillée des résidus
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, errors, alpha=0.5)
plt.xlabel('Prédictions')
plt.ylabel('Résidus')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Analyse des résidus')
plt.show()

# 9. Analyse des performances par plage de prix
df_results = pd.DataFrame({
    'Prix_reel': y_test,
    'Prix_predit': y_pred,
    'Erreur_absolue': abs(y_test - y_pred)
})

price_ranges = pd.qcut(df_results['Prix_reel'], q=5)
performance_by_range = df_results.groupby(price_ranges)['Erreur_absolue'].agg(['mean', 'std'])
print("\nPerformance par plage de prix:")
print(performance_by_range)

# 10. Prédictions détaillées
print("\nExemples de prédictions détaillées:")
sample_size = 5
sample_indices = np.random.choice(len(y_test), sample_size, replace=False)

for idx in sample_indices:
    print(f"\nÉchantillon {idx}:")
    print(f"Prix réel: {y_test[idx]:.2f}")
    print(f"Prix prédit: {y_pred[idx]:.2f}")
    print(f"Erreur absolue: {abs(y_test[idx] - y_pred[idx]):.2f}")