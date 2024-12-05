# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 15:44:48 2024

@author: Rana
"""

import numpy as np
import cv2
from skimage.io import imread

predicted_mask = cv2.imread("C:/Users/rana/anaconda3/envs/brain_tumor_detector/results/segmentation/Gaussian/GradientHistogramSeg/Cleaned_Segmented_gradHis.png", 0)  
ground_truth_mask = imread("C:/Users/rana/anaconda3/envs/brain_tumor_detector/data/lgg-mri-segmentation/kaggle_3m/TCGA_HT_7882_19970125/TCGA_HT_7882_19970125_12_mask.tif", as_gray=True) 


binary_mask = (predicted_mask > 127).astype(np.uint8)  
ground_truth_mask = (ground_truth_mask > 0.5).astype(np.uint8)

print("Unique values in binary predicted mask:", np.unique(binary_mask))
print("Unique values in ground truth mask:", np.unique(ground_truth_mask))


TP = np.sum((predicted_mask == 1) & (ground_truth_mask == 1))
FP = np.sum((predicted_mask == 1) & (ground_truth_mask == 0))
FN = np.sum((predicted_mask == 0) & (ground_truth_mask == 1))
TN = np.sum((predicted_mask == 0) & (ground_truth_mask == 0))
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


import matplotlib.pyplot as plt
plt.imshow(binary_mask, cmap='gray')
plt.title('Predicted Mask')
plt.show()

