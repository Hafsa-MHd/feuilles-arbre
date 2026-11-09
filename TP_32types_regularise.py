# Exercice 2 (variante) : Classification de 32 types de feuilles d'arbres
# CNN profond AVEC regularisation (data augmentation + dropout).
# Experience complementaire destinee a limiter le surapprentissage (memorisation
# de la base d'entrainement). Sur cette base, la regularisation reduit bien la
# memorisation (precision d'entrainement plus basse) mais n'ameliore pas la
# precision de validation, le jeu de validation etant tres proche de l'entrainement.

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers

size = 56
filter = 10
epoch = 30

# Generation de la base de donnees a partir des dossiers d'images.
train_ds, test_ds = keras.utils.image_dataset_from_directory(
    "data_32_classes",
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    validation_split=0.1,          # 10% pour la validation, 90% pour l'entrainement
    subset="both",
    seed=123,
    image_size=(size, size),
    batch_size=64,
)

# Noms des classes
class_names = train_ds.class_names
print("Classes :", class_names)

# Data augmentation : transformations aleatoires appliquees uniquement a
# l'entrainement pour limiter le surapprentissage.
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Construction de l'architecture profonde regularisee (4 blocs convolutifs + dropout)
model = keras.Sequential([
    data_augmentation,
    layers.Rescaling(1. / 255, input_shape=(size, size, 3)),   # normalisation des pixels
    layers.Conv2D(filter, (3, 3), strides=(1, 1), padding="same", activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(filter * 3, (3, 3), strides=(1, 1), padding="same", activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(filter * 25, (3, 3), strides=(1, 1), padding="same", activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(filter * 50, (3, 3), strides=(1, 1), padding="same", activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Flatten(),
    layers.Dropout(0.4),   # regularisation : desactive aleatoirement 40% des neurones
    layers.Dense(len(class_names), activation="softmax"),
])

# Compilation du modele
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Entrainement du modele
model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=epoch,
)

# Evaluation du modele
eval = model.evaluate(test_ds)

print("La precision du modele est de ", eval)

plt.show()
