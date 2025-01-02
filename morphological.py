# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 21:48:43 2025

@author: Rana
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'C:/Users/rana/anaconda3/envs/brain_tumor_detector/results2/medianski/watershed_medianski_highpass1.tif'  
binary_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


kernel_erosion = np.ones((5,5),np.uint8)   
eroded_image = cv2.erode(binary_image,kernel_erosion,iterations = 1)  

kernel_dilation = np.ones((3,3),np.uint8)
dilated_image = cv2.dilate(eroded_image,kernel_dilation,iterations = 1)



plt.figure(figsize=(15, 10))


plt.subplot(3, 1, 1)
plt.imshow(binary_image, cmap='gray')
plt.title('WaterShed Segmented Image')
plt.axis('off')

plt.subplot(3, 1, 2)
plt.imshow(eroded_image, cmap='gray')
plt.title('Eroded Image')
plt.axis('off')

plt.subplot(3, 1, 3)
plt.imshow(dilated_image, cmap='gray')
plt.title('Dilated Image')
plt.axis('off')

plt.tight_layout()
plt.show()



output_path = 'C:/Users/rana/anaconda3/envs/brain_tumor_detector/results2/medianski/dilated_medianski_highpass1.tif' 
cv2.imwrite(output_path, dilated_image)
output_path1 = 'C:/Users/rana/anaconda3/envs/brain_tumor_detector/results2/medianski/eroded_medianski_highpass1.tif'  
cv2.imwrite(output_path1, eroded_image)