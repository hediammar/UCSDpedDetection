import os
import cv2
import numpy as np
import glob
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Define the root directory
root_dir = r'C:\Users\21650\Downloads\UCSD_Anomaly_Dataset'  # Replace with your path

# Initialize dictionaries to store file paths
train_files = {'UCSDped1': [], 'UCSDped2': []}
test_files = {'UCSDped1': [], 'UCSDped2': []}

# Collect file paths for training and testing
for dataset in ['UCSDped1', 'UCSDped2']:
    train_dir = os.path.join(root_dir, dataset, 'Train')
    test_dir = os.path.join(root_dir, dataset, 'Test')

    # Collect training files
    train_paths = glob.glob(os.path.join(train_dir, '**', '*.[Tt][Ii][Ff]'), recursive=True)
    train_files[dataset].extend(train_paths)

    # Collect testing files
    test_paths = glob.glob(os.path.join(test_dir, '**', '*.[Tt][Ii][Ff]'), recursive=True)
    test_files[dataset].extend(test_paths)

# Function to preprocess images
def preprocess_images(image_paths, target_size=(128, 128)):
    images = []
    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Load image
        if img is not None:
            img = cv2.resize(img, target_size)  # Resize image
            img = img.astype('float32') / 255.0  # Normalize pixels
            images.append(img)
    return np.array(images)

# Preprocess training and test images for both datasets
train_images_ped1 = preprocess_images(train_files['UCSDped1'])
train_images_ped2 = preprocess_images(train_files['UCSDped2'])
test_images_ped1 = preprocess_images(test_files['UCSDped1'])
test_images_ped2 = preprocess_images(test_files['UCSDped2'])

# Combine images from both datasets
train_images = np.concatenate([train_images_ped1, train_images_ped2], axis=0)
test_images = np.concatenate([test_images_ped1, test_images_ped2], axis=0)

# Add channel dimension (for grayscale images)
train_images = train_images[..., np.newaxis]
test_images = test_images[..., np.newaxis]

# Define the Convolutional Autoencoder
def build_autoencoder(input_shape=(128, 128, 1)):
    # Encoder
    input_img = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    # Autoencoder model
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# Instantiate and summarize the autoencoder model
autoencoder = build_autoencoder()
autoencoder.summary()

# Train the autoencoder on normal frames only
epochs = 50
batch_size = 32

history = autoencoder.fit(
    train_images, train_images,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_split=0.1
)

# Set a threshold for anomaly detection based on training reconstruction error
reconstructed_train = autoencoder.predict(train_images)
mse_train = np.mean(np.square(train_images - reconstructed_train), axis=(1, 2, 3))
threshold = np.mean(mse_train) + 2 * np.std(mse_train)
print(f"Anomaly detection threshold: {threshold}")

# Detect anomalies in the test set
reconstructed_test = autoencoder.predict(test_images)
mse_test = np.mean(np.square(test_images - reconstructed_test), axis=(1, 2, 3))
anomalies = mse_test > threshold

# Display the number of anomalies detected
print(f"Number of anomalous frames detected: {np.sum(anomalies)}")

# Function to plot anomalous test images and their reconstructions
def plot_anomaly_detection(test_images, reconstructed_images, mse, threshold, num_images=5):
    plt.figure(figsize=(15, 6))
    anomalous_indices = np.where(mse > threshold)[0]
    for i, idx in enumerate(anomalous_indices[:num_images]):
        # Original
        plt.subplot(2, num_images, i + 1)
        plt.imshow(test_images[idx].squeeze(), cmap='gray')
        plt.title("Original (Anomalous)")
        plt.axis('off')

        # Reconstruction
        plt.subplot(2, num_images, i + num_images + 1)
        plt.imshow(reconstructed_images[idx].squeeze(), cmap='gray')
        plt.title(f"Reconstructed\nError: {mse[idx]:.4f}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Plot a few anomalies
plot_anomaly_detection(test_images, reconstructed_test, mse_test, threshold)
