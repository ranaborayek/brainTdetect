# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 00:26:00 2024

@author: Rana
"""

#%%Read an image%%
import cv2

def read_image(filepath):
    img = cv2.imread(filepath)
    if img is None:
        print("Error: Image not found!")
    return img

#%%Split channels and show them%% 
#We will work on FLAIR (second channel)
import matplotlib.pyplot as plt
def split_show_channels(image):
    Precontrast, FLAIR, Postcontrast = image[:, :, 0], image[:, :, 1], image[:, :, 2]
     # Display channels using Matplotlib
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(Precontrast)
    axes[0].set_title("Pre-contrast")
    axes[0].axis('off')

    axes[1].imshow(FLAIR)
    axes[1].set_title("FLAIR")
    axes[1].axis('off')
    
    axes[2].imshow(Postcontrast)
    axes[2].set_title("Post-contrast")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()
    return FLAIR

#%%Save FLAIR Channel%%
import os
def save_flair(FLAIR,output_dir):
 #Saving FLAIR sequence
   output_path = os.path.join(output_dir, "Flair.tif")
   cv2.imwrite(output_path, FLAIR)
   return output_path


#%%Apply gaussian%%
from skimage import img_as_float
from skimage import io
from scipy import ndimage as nd
from PIL import Image
def gaussian_filtering(noisy_img, output_dir):
 noisy_img = img_as_float(io.imread("C:/Users/rana//anaconda3/envs/brain_tumor_detector/results/Preprocessing/Flair.tif"))
 gaussian_img = nd.gaussian_filter(noisy_img , sigma=0.5)
 fig, ax = plt.subplots(1, 1, figsize=(15, 5))
 ax.imshow(gaussian_img)
 ax.set_title("Gaussian Filter, sigma=0.5")
 ax.axis('off')

 plt.tight_layout()
 plt.show()

 #Save filter
 path_gaussian = os.path.join(output_dir, "gaussian_filtered_image.tif")
 Image.fromarray(gaussian_img).save(path_gaussian)
 return path_gaussian



#%%Apply bilateral%%
from skimage import img_as_float
from skimage import io
from skimage.restoration import denoise_bilateral
import numpy as np
def bilateral_filtering(noisy_img, output_dir):
 noisy_img = img_as_float(io.imread("C:/Users/rana/anaconda3/envs/brain_tumor_detector/results/Preprocessing/Flair.tif"))
 bilateral_img = denoise_bilateral(noisy_img, sigma_spatial=20)
 fig, ax = plt.subplots(1, 1, figsize=(15, 5))
 ax.imshow(bilateral_img)
 ax.set_title("Bilateral Filter, sigma=20")
 ax.axis('off')

 plt.tight_layout()
 plt.show()

 #Save filter
 path_bilateral = os.path.join(output_dir, "gaussian_filtered_image.tif")
 Image.fromarray(bilateral_img).save(path_bilateral)
 return path_bilateral

#%%Canny%%

from skimage import data, img_as_ubyte, filters, feature 
from skimage.color import rgb2gray
def Canny(Filename):
#Canny edge detection
 img = cv2.imread(Filename,0)
 uint8_image = (img * 255).astype(np.uint8)
 
 canny_edge = cv2.Canny(uint8_image,50,80)

#Show images
 cv2.imshow("Original Image",canny_edge)
 cv2.imshow("Canny-edge filtered image",canny_edge)
 cv2.waitKey(0)
 cv2.destroyAllWindows()


#%%Main%%
if __name__ == "__main__":
    filepath = "C:/Users/rana/anaconda3/envs/brain_tumor_detector/data/lgg-mri-segmentation/kaggle_3m/TCGA_HT_7882_19970125/TCGA_HT_7882_19970125_12.tif"
    output_dir = "C:/Users/rana/anaconda3/envs/brain_tumor_detector/results/Preprocessing" 
    img = read_image(filepath)

    if img is not None:
      FLAIR = split_show_channels(img)
      
      OUTPUTtoFLAIR = save_flair(FLAIR,output_dir)
      
      Gaussian_path = gaussian_filtering(FLAIR,output_dir)
      Bilateral_path = bilateral_filtering(FLAIR,output_dir)
      
      Canny(Bilateral_path)
      Canny(Gaussian_path)
