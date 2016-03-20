# CS194-26: Computational Photography
# Project 1: Colorizing the Prokudin-Gorskii photo collection.
# colorize.py
# Krishna Parashar

import numpy as np
import skimage as sk
import skimage.io as skio
from scipy.misc import imresize

def split(img):
	# Splits the image into three equal images
	height = int(np.floor(img.shape[0] / 3.0))
	blue = img[:height]
	green = img[height: 2*height]
	red = img[2*height: 3*height]
	return blue, green, red

def crop(img):
	# Crops 10% off the borders of the image
	height, width = img.shape
	new_height, new_width = int(height/1.11), int(width/1.11)
	return img[height-new_height:new_height, width-new_width:new_width]

def ssd(a, b):
	# Sum of Square Differences: sum(sum((a-b).^2))
	return (((a-b)**2).sum()).sum()

def ncc(a, b):
	# Normalized Cross-Correlation: (a./||a|| dot b./||b||)
	return ((a/np.linalg.norm(a)) * (b/np.linalg.norm(b))).ravel().sum()

def find_displacement(img1, img2, bound, radius):
	# Searches over a radius in given images and returns best displacement
	best_score = -1
	for column in range(-radius + bound[0], radius + bound[0] ):
		for row in range(-radius + bound[1], radius + bound[1]):
			# print(np.shape(img2))
			img_displacement_score = ncc(img1, np.roll(np.roll(img2, column, axis = 0), row, axis = 1))
			if (img_displacement_score > best_score): 
				best_score = img_displacement_score
				img_displacement = [column, row]
	return img_displacement

def align(img, displacement):
	# Aligns images using appropriate displacement vectors
	aligned_img = np.roll(np.roll(img, displacement[0], axis = 0), displacement[1], axis = 1)
	return aligned_img

def naive_align(img1, img2, radius):
	# Controls flow for simple finding of displacement
	bound = [radius, radius]
	displacement = find_displacement(img1, img2, bound, radius)
	return displacement

def pyramid_align(img1, img2, radius):
	# Recursively scales down the image to an reasonable size
	rows, columns = img1.shape
	if max(rows, columns) > 200:
		img1_resized = imresize(img1, 0.5)
		img2_resized = imresize(img2, 0.5)
		courser_image_offset = pyramid_align(img1_resized, img2_resized, radius)
		scaled_courser_offset = [x*2 for x in courser_image_offset] 
		displacement = find_displacement(img1, img2, scaled_courser_offset, radius)
	else:
		displacement = naive_align(img1, img2, 15)
	return displacement

def stack_images(red_img, green_img, blue_img):
	# Overlays images using with RGB
	color_img = np.dstack([red_img, green_img, blue_img])
	return color_img

def white_balance(img):
	# Uses the gray world approach to balance the colors in the image
	pass

def write_image(dest_dir, img_name, img, extension):
	# Saves image to directory using the appropriate extension
	output_filename = dest_dir + img_name[:-4] + extension
	skio.imsave(str(output_filename), img)
	print("Image {0} saved to {1}".format(img_name, output_filename))
