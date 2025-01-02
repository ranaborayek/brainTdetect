# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 14:15:36 2025

@author: Rana
"""

import cv2
import numpy as np

# Load the original grayscale image
original_image = cv2.imread('C:/Users/rana/anaconda3/envs/brain_tumor_detector/data/lgg-mri-segmentation/kaggle_3m/TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_13.tif', cv2.IMREAD_GRAYSCALE)

# Ensure the high-pass filtered image exists (you should have already created this in a previous step)
# For example, if you created the high-pass filtered image earlier, you can load it like this:
high_pass_image = cv2.imread('C:/Users/rana/anaconda3/envs/brain_tumor_detector/results2/medianski/medianskimage_highpass10.tif', cv2.IMREAD_GRAYSCALE)

# Check if the images were loaded properly
if original_image is None or high_pass_image is None:
    print("Error: One or both images could not be loaded.")
else:
    # Normalize the high-pass filtered image to match the original image scale (if needed)
    high_pass_image_normalized = np.uint8((high_pass_image / np.max(high_pass_image)) * 255)

    # Add the high-pass filtered image to the original image
    enhanced_image = cv2.add(original_image, high_pass_image_normalized)

    # Save the enhanced image
    output_path = 'C:/Users/rana/anaconda3/envs/brain_tumor_detector/results2/medianski/enhanced_medianski_highpass10.tif'
    cv2.imwrite(output_path, enhanced_image)




