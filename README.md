# Classification de feuilles d'arbres

Mini-projet ENSEM. Des CNN sont comparés sur des photos de feuilles : d'abord 4 classes, puis 32.

Les notebooks enregistrent les courbes. Sur 4 classes, la validation dépasse l'objectif de 75 % (plusieurs essais au-dessus de 95 %). Sur 32 classes, le meilleur passage enregistré est autour de 84 % de validation.

Le compte-rendu écrit est `compte_rendu_final_feuilles.pdf`.

## Fichiers

- `farbre_proj4.ipynb` — 4 classes, dossier `./data_4_classes`
- `farbre_proj32.ipynb` — 32 classes, dossier `./data_32_classes`
- `compte_rendu_final_feuilles.pdf`
- `Code_Classe4_pdf.pdf` et `Code_Classe32_pdf.pdf` — export du code

## Données

Les images ne sont pas dans ce dépôt. Chaque classe est un sous-dossier de photos. Le jeu utilisé en cours est celui décrit par Munisami et al. (2015), *Plant Leaf Recognition Using Shape Features and Colour Histogram with K-nearest Neighbour Classifiers*.

```text
data_4_classes/
  espece_a/*.jpg
  espece_b/*.jpg
```

TensorFlow et scikit-learn sont nécessaires pour relancer l'entraînement.
