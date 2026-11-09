# Exercice 1 : Classification de 4 types de feuilles d'arbres (CNN, 1 bloc convolutif)

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers

size = 28
filter = 10
epoch = 30

# Generation de la base de donnees a partir des dossiers d'images.
# Le nom de chaque sous-dossier sert d'etiquette (label) de classe.
train_ds, test_ds = keras.utils.image_dataset_from_directory(
    "classe4/data_4_classes",
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

# Affichage d'une image d'exemple avec son etiquette
for images, labels in train_ds.take(1):
    plt.imshow(images[0].numpy().astype("uint8"))
    label_indice = np.argmax(labels[0])
    plt.title(class_names[label_indice])
    plt.axis("off")

# Construction de l'architecture pour des images RGB
model = keras.Sequential([
    layers.Rescaling(1. / 255, input_shape=(size, size, 3)),   # normalisation des pixels
    layers.Conv2D(
        filters=filter,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
        input_shape=(size, size, 3),   # 3 canaux car image RGB (et non noir et blanc)
    ),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Flatten(),
    layers.Dense(len(class_names), activation="softmax"),   # autant de sorties que de classes
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
