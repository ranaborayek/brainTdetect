# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 15:58:48 2024

@author: Rana
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import img_as_float, data, io, img_as_ubyte, filters, feature 
from skimage.color import rgb2gray
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.filters import threshold_multiotsu
from PIL import Image
import os
from scipy import ndimage as nd

 

'''try:
    img = Image.open('C:/Users/rana/anaconda3/envs/brain_tumor_detector/results/Preprocessing/gaussian_filtered_image.tif')
    img = np.array(img)  # Convert to NumPy array
except Exception as e:
    raise ValueError(f"Error loading image: {e}")

if img is None:
    raise ValueError("Error: Image not loaded. Check the file path.")

if img.dtype not in [np.float64, np.float32]:
    raise TypeError(f"Expected a float64 or float32 image, but got {img.dtype}!")


if img.dtype != np.uint8:
    if img.max() <= 1:  
        img = (img * 255).astype(np.uint8)
    else:  
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)


if len(img.shape) == 3: 
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Convert to grayscale

#used for debugging
#print("Image dtype:", img.dtype)
#print("Image min:", img.min(), "Image max:", img.max())


canny_edge = cv2.Canny(img,0,48)

#Auto_canny
sigma = 5
median = np.median(img)
lower = int(max(0, (1.0 - sigma) * median))
upper = int(min(255, (1.0 + sigma) * median))
auto_canny = cv2.Canny(img, lower, upper)

#plot
fig1, ax1 = plt.subplots()
ax1.imshow(img, cmap='gray')
ax1.set_title("Original Image")
ax1.axis('off') 
output_path = 'C:/Users/rana/anaconda3/envs/brain_tumor_detector/results/Segmentation/bilateral results/original_image.tif'
fig1.canvas.draw()  
img_output = np.array(fig1.canvas.renderer.buffer_rgba())  
img = Image.fromarray(img_output)  
img = img.convert("RGB")  
img.save(output_path, format="TIFF")

fig2, ax2 = plt.subplots()
ax2.imshow(canny_edge, cmap='gray')
ax2.set_title("Canny Edge Image")
ax2.axis('off') 
output_path = 'C:/Users/rana/anaconda3/envs/brain_tumor_detector/results/Segmentation/bilateral results/Canny_edge_output.tif'
fig2.canvas.draw()  
img_output = np.array(fig2.canvas.renderer.buffer_rgba())  
img = Image.fromarray(img_output)  
img = img.convert("RGB")  
img.save(output_path, format="TIFF")
plt.show()

#show

cv2.imshow("Original Image", img)
cv2.waitKey(0)

Image.fromarray(img).save("original_image.tif")

cv2.imshow("Canny Edge-Detected Image", canny_edge)
cv2.waitKey(0)

Image.fromarray(canny_edge).save("canny_edge_image.tif")

cv2.destroyAllWindows()


output_dir = "C:/Users/rana/anaconda3/envs/brain_tumor_detector/results/segmentation/Gaussian"

Image.fromarray(img).save(os.path.join(output_dir, "original.tif"))
Image.fromarray(canny_edge).save(os.path.join(output_dir, "canny_edge_image.tif"))
'''

'''#MultiOtsu
image = io.imread("C:/Users/rana/anaconda3/envs/brain_tumor_detector/results/segmentation/bilateral results/canny_edge_image.tif")

thresholds = threshold_multiotsu(image,classes=2)

regions = np.digitize (image, bins=thresholds)

output = img_as_ubyte(regions)

output_path = "C:/Users/rana/anaconda3/envs/brain_tumor_detector/results/segmentation/bilateral results/bilateral_otsu_segmented.tif"
output_image = Image.fromarray(output)
output_image.save(output_path, format="TIFF")

fig,ax = plt.subplots(1,3, figsize=(15,7))

ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original')
ax[0].axis('off')

ax[1].hist(image.ravel(),bins=255)
ax[1].set_title('Histogram')

for thresh in thresholds:
    ax[1].axvline(thresh, color='r')

ax[2].imshow(regions, cmap='Accent')
ax[2].set_title('Multi-Otsu result')
ax[2].axis('off')

plt.subplots_adjust()

plt.show()'''
'''
#OTSU Thresholding, binarization

img = cv2.imread("C:/Users/rana/anaconda3/envs/brain_tumor_detector/results/segmentation/bilateral results/canny_edge_image.tif", 0)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  
clahe_img = clahe.apply(img)

plt.hist(clahe_img.flat, bins =100, range=(0,255))

# binary thresholding
ret1,th1 = cv2.threshold(clahe_img,185,200,cv2.THRESH_BINARY)

# Otsu's thresholding
ret2,th2 = cv2.threshold(clahe_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


cv2.imshow("Binarization", th1)
cv2.waitKey(0)
cv2.imshow("Otsu", th2)
output_dir = "C:/Users/rana/anaconda3/envs/brain_tumor_detector/results/segmentation/bilateral results"
Image.fromarray(th2).save(os.path.join(output_dir, "otsu.tif"))
cv2.waitKey(0)          
cv2.destroyAllWindows() 


#thresholdingandmorphologicaloperation
img = cv2.imread("C:/Users/rana/anaconda3/envs/brain_tumor_detector/results/segmentation/Gaussian/canny_edge_image.tif", 0)

ret,th = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

kernel = np.ones((3,3),np.uint8)   # 3x3 kernel with all ones. 

gradient = cv2.morphologyEx(th, cv2.MORPH_GRADIENT, kernel)



cv2.imshow("Original Image", img)
cv2.imshow("Otsu",th)
cv2.imshow("gradient",gradient)
cv2.waitKey(0)          
cv2.destroyAllWindows() 

output_dir = "C:/Users/rana/anaconda3/envs/brain_tumor_detector/results/segmentation/Gaussian"
Image.fromarray(gradient).save(os.path.join(output_dir, "gradient.tif"))
cv2.waitKey(0)    
'''
      
#Histogram based Segmentation

img = io.imread("C:/Users/rana/anaconda3/envs/brain_tumor_detector/results/segmentation/bilateral results/gradient.tif")

float_img = img_as_float(img)
sigma_est = np.mean(estimate_sigma(float_img))


denoise_img = denoise_nl_means(float_img, h=1.15 * sigma_est, fast_mode=False, 
                               patch_size=5, patch_distance=3)
                           
denoise_img_as_8byte = img_as_ubyte(denoise_img)

plt.imshow(denoise_img_as_8byte, cmap=plt.cm.gray, interpolation='nearest')
cv2.imshow("Denoised Image", denoise_img_as_8byte)



segm1 = (denoise_img_as_8byte <= 57)
segm2 = (denoise_img_as_8byte > 57) & (denoise_img_as_8byte <= 110)
segm3 = (denoise_img_as_8byte > 110) & (denoise_img_as_8byte <= 210)
segm4 = (denoise_img_as_8byte > 210)

all_segments = np.zeros((denoise_img_as_8byte.shape[0], denoise_img_as_8byte.shape[1], 3)) #nothing but denoise img size but blank

all_segments[segm1] = (1,0,0)
all_segments[segm2] = (0,1,0)
all_segments[segm3] = (0,0,1)
all_segments[segm4] = (1,1,0)

cv2.imshow("Segmented Image", all_segments)


segm1_opened = nd.binary_opening(segm1, np.ones((3,3)))
segm1_closed = nd.binary_closing(segm1_opened, np.ones((3,3)))

segm2_opened = nd.binary_opening(segm2, np.ones((3,3)))
segm2_closed = nd.binary_closing(segm2_opened, np.ones((3,3)))

segm3_opened = nd.binary_opening(segm3, np.ones((3,3)))
segm3_closed = nd.binary_closing(segm3_opened, np.ones((3,3)))

segm4_opened = nd.binary_opening(segm4, np.ones((3,3)))
segm4_closed = nd.binary_closing(segm4_opened, np.ones((3,3)))




all_segments_cleaned = np.zeros((denoise_img_as_8byte.shape[0], denoise_img_as_8byte.shape[1], 3)) 


all_segments_cleaned[segm1_closed] = (1,0,0)
all_segments_cleaned[segm2_closed] = (0,1,0)
all_segments_cleaned[segm3_closed] = (0,0,1)
all_segments_cleaned[segm4_closed] = (1,1,0)

cv2.imshow("Cleaned Segmented Image", all_segments_cleaned)

cv2.waitKey(0)
cv2.destroyAllWindows()
















