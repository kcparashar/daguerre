# CS194-26: Computational Photography
# Project 3: Fun with Frequencies
# unsharp.py (Part 0)
# Krishna Parashar


import numpy as np
import skimage as sk
import skimage.io as skio
import matplotlib.pyplot as plt
from skimage.filters import gaussian_filter
from skimage.exposure import rescale_intensity


def gaussian(img, sigma):
	# Performs a low pass filter operation in the image
	return gaussian_filter(img, sigma, mode='reflect')

def laplacian(img, sigma):
	# Takes the Laplacian of an image which leaves the high frequency edges
	return img - gaussian(img, sigma)

def unsharp(img, sigma, alpha):
	# Sharpens image by adding the high frequency edges to the original image
	sharp_img = img + (alpha * (laplacian(img, sigma)))
	return rescale_intensity(sharp_img, in_range=(0, 1), out_range=(0, 1))

def write_image(dest_dir, img_name, img, attribute, extension):
	# Saves image to directory using the appropriate extension
	output_filename = dest_dir + img_name[:-4] + attribute + extension
	plt.imsave(str(output_filename), img)
	print("Image {0} saved to {1}".format(img_name, output_filename))
