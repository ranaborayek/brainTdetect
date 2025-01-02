# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 20:18:53 2025

@author: Rana
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


image_path = 'C:/Users/rana/anaconda3/envs/brain_tumor_detector/results2/cv2filter/thresh_cv2filter_highpass4.tif'  
binary_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2,5)


_, foreground = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
foreground = np.uint8(foreground)  

background = cv2.dilate(binary_image, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=3)

unknown = cv2.subtract(background, foreground)

_, markers = cv2.connectedComponents(foreground)
markers = markers + 1
markers[unknown == 255] = 0  


binary_rgb = cv2.merge((binary_image, binary_image, binary_image))
markers = cv2.watershed(binary_rgb, markers)

segmented_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

segmented_image[markers == -1] = [255, 0, 0] 

plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
plt.title('Watershed Segmentation')
plt.axis('off')
plt.show()


output_path = 'C:/Users/rana/anaconda3/envs/brain_tumor_detector/results2/cv2filter/watershed_cv2filter_highpass4.tif'  
cv2.imwrite(output_path, binary_image)
