import os
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import MeanSquaredError
from data_preprocessing import train_images, test_images  # Import the images (use this if you pre-process them in data_preprocessing.py)

# Check if images are loaded properly
print("Train images shape:", train_images.shape)
print("Test images shape:", test_images.shape)

# Autoencoder model definition
input_img = Input(shape=(128, 128, 1))

# Encoder
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

# Define the model
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer=Adam(), loss=MeanSquaredError())

# Train the autoencoder
autoencoder.fit(train_images, train_images, epochs=20, batch_size=128, shuffle=True, validation_split=0.1)

# Save the trained model
autoencoder.save('autoencoder_model.h5')
print("Model saved as 'autoencoder_model.h5'")

# Predict reconstructions on test set
reconstructions = autoencoder.predict(test_images)

# Calculate Mean Squared Error (MSE) for each image
mse = np.mean(np.power(test_images - reconstructions, 2), axis=(1, 2, 3))

# Print the MSE for each image
print("MSE for each image in the test set:")
print(mse)

# Function to visualize original and reconstructed images
def visualize_reconstruction(model, images, num=5):
    reconstructed = model.predict(images)
    plt.figure(figsize=(15, 5))
    for i in range(num):
        plt.subplot(2, num, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.axis('off')
        
        plt.subplot(2, num, num + i + 1)
        plt.imshow(reconstructed[i].squeeze(), cmap='gray')
        plt.axis('off')
    plt.show()

# Visualize some original and reconstructed test samples
visualize_reconstruction(autoencoder, test_images)
