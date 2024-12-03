import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 1. Génération des données
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

# 2. Application de K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# 3. Visualisation des clusters
plt.figure(figsize=(10, 6))

# Tracer les points et colorer selon le cluster
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50, alpha=0.6)

# Tracer les centres des clusters
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centres des clusters')

# Personnalisation du graphique
plt.title("Clusters formés par K-Means")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# 4. Analyse des résultats
unique, counts = np.unique(labels, return_counts=True)
cluster_counts = dict(zip(unique, counts))

print("Nombre de points par cluster :")
for cluster_id, count in cluster_counts.items():
    print(f"Cluster {cluster_id}: {count} points")

print("\nCentres des clusters :")
for i, center in enumerate(centers):
    print(f"Cluster {i}: {center}")

print("------------------------------------------------")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Génération des données
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

# Fonction pour tracer les clusters
def plot_kmeans(X, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50, alpha=0.6)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centres des clusters')
    plt.title(f"K-Means avec k={k}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

# Création des sous-graphiques pour k=2, k=3, k=4
plt.figure(figsize=(18, 5))

for i, k in enumerate([2, 3, 4], 1):
    plt.subplot(1, 3, i)
    plot_kmeans(X, k)

plt.tight_layout()
plt.show()

print("------------------------------------------------")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Création du dataset
np.random.seed(42)

# Générer des données réalistes
n_clients = 200
ages = np.concatenate([
    np.random.normal(25, 5, 70),    # Jeunes
    np.random.normal(45, 8, 80),    # Âge moyen
    np.random.normal(65, 5, 50)     # Seniors
])

depenses = np.concatenate([
    np.random.normal(500, 150, 70),  # Dépenses modérées
    np.random.normal(1000, 300, 80), # Dépenses moyennes
    np.random.normal(1500, 400, 50)  # Dépenses élevées
])

# Créer le DataFrame
df = pd.DataFrame({
    'Age': ages,
    'Depenses': depenses
})

# 2. Préparation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 3. Application de K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 4. Visualisation
plt.figure(figsize=(12, 8))

# Créer un scatter plot avec une couleur différente pour chaque cluster
scatter = plt.scatter(df['Age'], df['Depenses'], 
                     c=df['Cluster'], 
                     cmap='viridis',
                     s=100,
                     alpha=0.6)

# Ajouter les centres des clusters
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], 
           c='red',
           marker='X',
           s=200,
           label='Centres des clusters')

# Personnalisation du graphique
plt.title('Segmentation des Clients par Âge et Dépenses', fontsize=14)
plt.xlabel('Âge', fontsize=12)
plt.ylabel('Dépenses (€)', fontsize=12)
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.grid(True, linestyle='--', alpha=0.7)

plt.show()

# 5. Analyse des clusters
print("\nAnalyse des clusters :")
for cluster in range(3):
    cluster_data = df[df['Cluster'] == cluster]
    print(f"\nCluster {cluster}:")
    print(f"Nombre de clients: {len(cluster_data)}")
    print(f"Âge moyen: {cluster_data['Age'].mean():.1f} ans")
    print(f"Dépenses moyennes: {cluster_data['Depenses'].mean():.0f} €")
    print(f"Écart-type âge: {cluster_data['Age'].std():.1f}")
    print(f"Écart-type dépenses: {cluster_data['Depenses'].std():.0f} €")