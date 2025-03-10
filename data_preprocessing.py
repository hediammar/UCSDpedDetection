import os
import cv2
import numpy as np
import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Définir le répertoire racine
root_dir = r'C:\Users\21650\Downloads\UCSD_Anomaly_Dataset'  # Remplacez par votre chemin

# Initialiser des dictionnaires pour stocker les chemins des fichiers
train_files = {'UCSDped1': [], 'UCSDped2': []}
test_files = {'UCSDped1': [], 'UCSDped2': []}

# Itérer à travers UCSDped1 et UCSDped2 pour collecter les chemins des fichiers
for dataset in ['UCSDped1', 'UCSDped2']:
    # Fichiers d'entraînement
    train_dir = os.path.join(root_dir, dataset, 'Train')
    print(f"Checking directory: {train_dir}")
    train_paths = glob.glob(os.path.join(train_dir, '**', '*.[Tt][Ii][Ff]'), recursive=True)  # Recherche récursive
    print(f"Found {len(train_paths)} training files in {train_dir}")
    train_files[dataset].extend(train_paths)

    # Fichiers de test
    test_dir = os.path.join(root_dir, dataset, 'Test')
    print(f"Checking directory: {test_dir}")
    test_paths = glob.glob(os.path.join(test_dir, '**', '*.[Tt][Ii][Ff]'), recursive=True)  # Recherche récursive
    print(f"Found {len(test_paths)} testing files in {test_dir}")
    test_files[dataset].extend(test_paths)

# Fonction de prétraitement des images
def preprocess_images(image_paths, target_size=(128, 128)):
    images = []
    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Charger l'image
        if img is not None:
            # Redimensionner l'image
            img = cv2.resize(img, target_size)
            # Normaliser les pixels entre 0 et 1
            img = img.astype('float32') / 255.0
            images.append(img)
    return np.array(images)

# Appliquer le prétraitement aux images d'entraînement et de test pour UCSDped1 et UCSDped2
train_images_ped1 = preprocess_images(train_files['UCSDped1'])  # Images d'entraînement UCSDped1
test_images_ped1 = preprocess_images(test_files['UCSDped1'])    # Images de test UCSDped1
train_images_ped2 = preprocess_images(train_files['UCSDped2'])  # Images d'entraînement UCSDped2
test_images_ped2 = preprocess_images(test_files['UCSDped2'])    # Images de test UCSDped2

# Ajouter une dimension pour les canaux (1 pour les images en niveaux de gris)
train_images_ped1 = train_images_ped1[..., np.newaxis]
test_images_ped1 = test_images_ped1[..., np.newaxis]
train_images_ped2 = train_images_ped2[..., np.newaxis]
test_images_ped2 = test_images_ped2[..., np.newaxis]

# Combiner les ensembles d'entraînement et de test de UCSDped1 et UCSDped2
train_images = np.concatenate((train_images_ped1, train_images_ped2), axis=0)
test_images = np.concatenate((test_images_ped1, test_images_ped2), axis=0)

# Afficher le nombre total d'images d'entraînement et de test
print(f"Nombre total d'images d'entraînement : {len(train_images)}")
print(f"Nombre total d'images de test : {len(test_images)}")


# Fonction d'augmentation des données
def augment_data(images):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    return datagen.flow(images, batch_size=32)

# Exemple d'utilisation de l'augmentation des données
augmented_train_data = augment_data(train_images)

# Afficher le nombre d'images d'entraînement et de test
print(f"Nombre d'images d'entraînement : {len(train_images)}")
print(f"Nombre d'images de test : {len(test_images)}")

# Fonction pour afficher quelques images
def plot_images(images, num=5):
    plt.figure(figsize=(15, 5))
    for i in range(num):
        plt.subplot(1, num, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')  # Utiliser cmap='gray' pour les images en niveaux de gris
        plt.axis('off')
    plt.show()

# Afficher quelques images d'entraînement
plot_images(train_images)