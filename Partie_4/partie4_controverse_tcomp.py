import numpy as np
import matplotlib.pyplot as plt

# Création des grilles de x et y
X, Y = np.meshgrid(np.linspace(1, 50000, 500), np.linspace(1, 512, 200))

# Calcul de la fonction
F = (1024 * 32) / (Y * (32 + np.log2(X)))

# Création du graphique avec un contour plot
plt.figure(figsize=(10, 6))

# Tracé des contours avec une bordure pour f(x, y) ≤ 2
contour = plt.contourf(X, Y, F, levels=50, cmap="viridis")
contour_lines = plt.contour(X, Y, F, levels=[5], colors="red", linewidths=2)

# Ajout de la barre de couleur
plt.colorbar(contour, label="f(x, y)")

# Personnalisation du graphique
plt.xlabel("x")
plt.ylabel("y")
plt.title("Tracé de f(x, y) avec bordure pour f(x) ≤ 5")
plt.xscale("log")  # Échelle logarithmique pour x
plt.yscale("log")  # Échelle logarithmique pour y
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Affichage du graphique
plt.show()
