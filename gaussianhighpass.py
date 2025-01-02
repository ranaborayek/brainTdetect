# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 12:25:04 2025

@author: Rana
"""

# libraries
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


save_directory = r"C:/Users/rana/anaconda3/envs/brain_tumor_detector/results2"
f = cv2.imread('C:/Users/rana/anaconda3/envs/brain_tumor_detector/results2/medianskimage.tif',0)

# open the image f

# transform the image into frequency domain, f --> F
F = np.fft.fft2(f)
Fshift = np.fft.fftshift(F)


# Create Gaussin Filter: Low Pass Filter
M,N = f.shape
H = np.zeros((M,N), dtype=np.float32)
D0 = 4
for u in range(M):
    for v in range(N):
        D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
        H[u,v] = np.exp(-D**2/(2*D0*D0))



# Gaussian: High pass filter
HPF = 1 - H


# Image Filters
Gshift = Fshift * HPF
G = np.fft.ifftshift(Gshift)
g = np.abs(np.fft.ifft2(G))


g_normalized = (g / np.max(g) * 255).astype(np.uint8)


cv2.imshow("high pass", (g_normalized / np.max(g) * 255).astype(np.uint8))


plt.imshow(g_normalized, cmap='gray')
plt.axis('off')
plt.show()

# Save the result using Pillow
output_path = os.path.join(save_directory, f"medianskimage_highpass{D0}.tif")
highpass_image = Image.fromarray(g_normalized)
highpass_image.save(output_path, format='TIFF')


cv2.waitKey(0)          
cv2.destroyAllWindows() 
