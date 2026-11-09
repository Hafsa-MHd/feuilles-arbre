# Classification de feuilles d'arbres

Compte rendu du 30 juin 2026, ENSEM, méthodes d'apprentissage. Hafsa Mehdaoui.

Deux CNN classent des photos de feuilles. Le seuil demandé était 75 % de précision de validation.

| Base | Architecture retenue | Précision de validation |
| --- | --- | --- |
| 4 espèces | 1 bloc convolutif, images 28×28, 30 epochs | 98,4 % |
| 32 espèces | 4 blocs convolutifs, images 56×56, 30 epochs | 96,8 % |

Une variante avec augmentation des images et dropout (40 %) a été testée sur les 32 espèces. Elle limite la mémorisation de l'entraînement, mais la validation tombe à 83,8 %. Ce n'est pas le modèle retenu.

Le détail est dans [cr_feuilles_HafsaMehdaoui.pdf](cr_feuilles_HafsaMehdaoui.pdf).

## Programmes

- `TP_4types.py` — 4 espèces. Les images sont dans `classe4/data_4_classes/`, une espèce par sous-dossier.
- `TP_32types.py` — 32 espèces. Les images sont dans `data_32_classes/`.
- `TP_32types_regularise.py` — la même base, avec augmentation et dropout.

Les photos ne sont pas dans ce dépôt. TensorFlow et Matplotlib suffisent pour relancer un script.
