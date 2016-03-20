# CS194-26: Computational Photography
# Final Project: Faking Tilt Shifts to Make Image Miniatures
# miniatures.py
# Krishna Parashar

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from skimage.filters import gaussian_filter
from skimage.exposure import rescale_intensity

def gaussian(img, sigma):
	# Performs a low pass filter operation in the image
	return gaussian_filter(img, sigma, mode='reflect')

def laplacian(img, sigma):
	# Takes the Laplacian of an image which leaves the high frequency edges
	return img - gaussian(img, sigma)

def gaussian_stack(img, levels):
    gaussian_stack = []
    sigma_list = [2**value for value in range(levels)]
    for sigma in sigma_list:
        gaussian_stack.append(gaussian(img, sigma))
    return gaussian_stack

def laplacian_stack(img, levels, gaussian_stack):
    laplacian_stack = []
    for item in gaussian_stack:
        laplacian_stack.append(img - item)
    return laplacian_stack

def mask_img(img, mask, side):
    if (side == "left") or (side == "top"): 
        multiplier = mask
    elif (side == "right") or (side == "bottom"): 
        multiplier = 1 - mask

    channels = []
    channels.append(img[:, :, 0] * multiplier)
    channels.append(img[:, :, 1] * multiplier)
    channels.append(img[:, :, 2] * multiplier)
    
    return np.dstack(channels)

def mask_imgs(img_stack, mask_stack, side):
    masked_stack = []
    for img, mask in zip(img_stack, mask_stack):
        masked_stack.append(mask_img(img, mask, side))
    return masked_stack

def blend(img1, blurred_img, img1_side, blurred_img_side, img_stack, mask, levels):
    sigma = 2**(levels)
    gaussian_mask = gaussian(mask, sigma)
    img1_low_freq = mask_img(gaussian(img1, sigma), gaussian_mask, img1_side)
    blurred_img_low_freq = mask_img(gaussian(blurred_img, sigma), gaussian_mask, blurred_img_side)
    return sum(img_stack, sum([img1_low_freq, blurred_img_low_freq]))

def write_image(dest_dir, img_name, img, attribute, extension):
    # Saves image to directory using the appropriate extension
    output_filename = dest_dir + img_name + attribute + extension
    cv2.imwrite(str(output_filename), img)
    print("Image {0} saved to {1}".format(img_name, output_filename))
