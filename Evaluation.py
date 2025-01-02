# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 15:44:48 2024

@author: Rana
"""


import numpy as np
import cv2
from skimage.io import imread

# Load the predicted mask (8-bit image)
predicted_mask = cv2.imread("C:/Users/rana/anaconda3/envs/brain_tumor_detector/results2/medianski/dilated_medianski_highpass1.tif", 0)

# Load the ground truth mask (binary mask with values 0 or 1)
ground_truth_mask = imread("C:/Users/rana/anaconda3/envs/brain_tumor_detector/data/lgg-mri-segmentation/kaggle_3m/TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_13_mask.tif", as_gray=True)

# Convert predicted mask to binary: values > 127 as 1 (tumor), else 0 (background)
binary_mask = (predicted_mask > 127).astype(np.uint8)

# Convert ground truth mask to binary (0 or 1)
ground_truth_mask = (ground_truth_mask > 0.5).astype(np.uint8)

# Now calculate metrics
TP = np.sum((binary_mask == 1) & (ground_truth_mask == 1))
FP = np.sum((binary_mask == 1) & (ground_truth_mask == 0))
FN = np.sum((binary_mask == 0) & (ground_truth_mask == 1))
TN = np.sum((binary_mask == 0) & (ground_truth_mask == 0))

print("TP:", TP)
print("FP:", FP)
print("FN:", FN)
print("TN:", TN)

accuracy = (TP + TN) / (TP + TN + FP + FN)
dice = (2 * TP) / (2 * TP + FP + FN)
sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

print(f"Accuracy: {accuracy:.4f}")
print(f"Dice Score: {dice:.4f}")
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Jaccard Index (IoU): {iou:.4f}")

