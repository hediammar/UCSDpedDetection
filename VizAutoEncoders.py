import numpy as np
import math
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.models import load_model
from data_preprocessing import train_images, test_images  # Import the images (use this if you pre-process them in data_preprocessing.py)
from tensorflow.keras.layers import Input

# Check if images are loaded properly
print("Train images shape:", train_images.shape)
print("Test images shape:", test_images.shape)

# Autoencoder model definition
input_img = Input(shape=(128, 128, 1))

# Load the saved autoencoder model
autoencoder = load_model('autoencoder_model.h5',custom_objects={'mse': losses.MeanSquaredError()})
print("Model loaded successfully.")

# Predict reconstructions on the test set
reconstructions = autoencoder.predict(test_images)

# Calculate Mean Squared Error (MSE) for each image
mse = np.mean(np.power(test_images - reconstructions, 2), axis=(1, 2, 3))

# Calculate SSIM for each image
ssim_values = []
for i in range(test_images.shape[0]):
    ssim_value = ssim(test_images[i].squeeze(), reconstructions[i].squeeze())
    ssim_values.append(ssim_value)

# Calculate PSNR for each image
psnr_values = []
for i in range(test_images.shape[0]):
    mse_value = np.mean(np.square(test_images[i] - reconstructions[i]))  # Compute MSE for PSNR
    if mse_value == 0:
        psnr = 100  # If there is no error, PSNR is infinite
    else:
        psnr = 20 * math.log10(1.0 / math.sqrt(mse_value))  # Assuming pixel values are in [0, 1]
    psnr_values.append(psnr)

# Print the MSE, SSIM, and PSNR for each image
print("MSE for each image in the test set:")
print(mse)
print("SSIM values for each image in the test set:")
print(ssim_values)
print("PSNR values for each image in the test set:")
print(psnr_values)

# Visualize some original and reconstructed test samples
def visualize_reconstruction(model, images, num=5):
    reconstructed = model.predict(images)
    plt.figure(figsize=(15, 5))
    for i in range(num):
        # Plot the original images
        plt.subplot(2, num, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.axis('off')
        
        # Plot the reconstructed images
        plt.subplot(2, num, num + i + 1)
        plt.imshow(reconstructed[i].squeeze(), cmap='gray')
        plt.axis('off')
    plt.show()

# Visualize some original and reconstructed test samples
visualize_reconstruction(autoencoder, test_images)
