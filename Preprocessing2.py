# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 20:44:55 2024

@author: Rana
"""

import cv2
import os
import numpy as np
#from scipy.ndimage.filters import convolve
#from skimage import io
from skimage.filters import gaussian
from skimage.filters import median
from PIL import Image


#img = io.imread('C:/Users/rana/anaconda3/envs/brain_tumor_detector/data/lgg-mri-segmentation/kaggle_3m/TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_13.tif', as_gray=True)
save_directory = r"C:/Users/rana/anaconda3/envs/brain_tumor_detector/results2"


os.makedirs(save_directory, exist_ok=True)

#Needs 8 bit, not float.
img_gaussian_noise = cv2.imread('C:/Users/rana/anaconda3/envs/brain_tumor_detector/data/lgg-mri-segmentation/kaggle_3m/TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_13.tif', 0)
img_salt_pepper_noise = cv2.imread('C:/Users/rana/anaconda3/envs/brain_tumor_detector/data/lgg-mri-segmentation/kaggle_3m/TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_13.tif', 0)

img = img_salt_pepper_noise

img1 = img_gaussian_noise

gaussian_kernel = np.array([[1/16, 1/8, 1/16],   #3x3 kernel
                [1/8, 1/4, 1/8],
                [1/16, 1/8, 1/16]])



conv_using_cv2 = cv2.filter2D(img1, -1, gaussian_kernel, borderType=cv2.BORDER_CONSTANT) 
# when ddepth=-1, the output image will have the same depth as the source
#example, if input is float64 then output will also be float64
# BORDER_CONSTANT - Pad the image with a constant value (i.e. black or 0)
#BORDER_REPLICATE: The row or column at the very edge of the original is replicated to the extra border.

gaussian_using_cv2 = cv2.GaussianBlur(img1, (3,3), 0, borderType=cv2.BORDER_CONSTANT)

gaussian_using_skimage = gaussian(img, sigma=1, mode='constant', cval=0.0)
#sigma defines the std dev of the gaussian kernel. SLightly different than 
#how we define in cv2


print(img_salt_pepper_noise.dtype)
print(img_gaussian_noise.dtype)

median_using_cv2 = cv2.medianBlur(img, 3)

from skimage.morphology import disk
median_using_skimage = median(img, disk(3), mode='constant', cval=0.0)


cv2.imshow("Original", img)
cv2.imshow("cv2 filter", conv_using_cv2)
cv2.imshow("Using cv2 gaussian", gaussian_using_cv2)
cv2.imshow("Using skimage", gaussian_using_skimage)
cv2.imshow("cv2 median", median_using_cv2)
cv2.imshow("Using skimage median", median_using_skimage)

# Save images in .tif format
cv2.imwrite(os.path.join(save_directory, "cv2filter.tif"), conv_using_cv2)
cv2.imwrite(os.path.join(save_directory, "cv2gaussian.tif"), gaussian_using_cv2)
cv2.imwrite(os.path.join(save_directory, "cv2median.tif"), median_using_cv2)
cv2.imwrite(os.path.join(save_directory, "medianskimage.tif"), median_using_skimage)


output_path = os.path.join(save_directory, "gaussianskimage.tif")
highpass_image = Image.fromarray(gaussian_using_skimage)
highpass_image.save(output_path, format='TIFF')




cv2.waitKey(0)          
cv2.destroyAllWindows() 
