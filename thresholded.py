# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 13:59:40 2025

@author: Rana
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the grayscale image
image_path = 'C:/Users/rana/anaconda3/envs/brain_tumor_detector/results2/medianski/enhanced_medianski_highpass10.tif'  # Change this to your image path
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded properly
if original_image is None:
    print("Error: Image could not be loaded.")
else:
    # Set a threshold value (you can experiment with different values)
    threshold_value = 150 # Adjust this threshold as needed

    # Apply thresholding using OpenCV's threshold function
    _, binary_image = cv2.threshold(original_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Display the original and binary images side by side
    plt.figure(figsize=(10, 5))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Binary (Thresholded) Image
    plt.subplot(1, 2, 2)
    plt.imshow(binary_image, cmap='gray')
    plt.title('Thresholded Image (Binary)')
    plt.axis('off')

    plt.show()

    
    output_path = 'C:/Users/rana/anaconda3/envs/brain_tumor_detector/results2/medianski/thresh_medianski_highpass10.tif'  # You can change the output path or filename
    cv2.imwrite(output_path, binary_image)

