# CS194-26: Computational Photography
# Project 6: Lightfield Camera
# aperture.py
# Krishna Parashar

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def read_images(source_dir, folder, extension):
	# Reads images from folder and saves to array
	print ("Loading dataset {0} from {1}".format(folder[:-1], (source_dir + folder)))
	imgs = []
	filenames = glob.glob(source_dir + folder + "*" + extension)
	for filename in filenames:
		imgs.append(plt.imread(filename))
	return imgs

def adjust_aperture(imgs, aperture_radius):
	# Produces an images with aperture radius surrounding the center image
	print ("Adjusting aperture for image using radius of {0}".format(aperture_radius))
	assert (aperture_radius >= 0)
	aperture_radius = aperture_radius % 8

	ref_array = np.arange(289).reshape((17, 17))
	center_coord = (np.floor(ref_array.shape[0]/2.), np.ceil(ref_array.shape[0]/2.))
	selected_imgs = np.array(ref_array[center_coord[0] - aperture_radius: center_coord[1] + aperture_radius, 
									   center_coord[0] - aperture_radius: center_coord[1] + aperture_radius]).flatten()
	adjusted_img = np.mean([imgs[number] for number in selected_imgs], axis=0)
	return adjusted_img

def write_images(dest_dir, img_name, img, attribute, counter, extension):
	# Saves image to directory using the appropriate extension
	output_filename = dest_dir + img_name + attribute + counter + extension
	plt.imsave(str(output_filename), img)
	print("Image {0} saved to {1}".format(img_name, output_filename))
	