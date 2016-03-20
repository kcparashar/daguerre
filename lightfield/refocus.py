# CS194-26: Computational Photography
# Project 6: Lightfield Camera
# refocus.py
# Krishna Parashar

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def refocus(imgs, scale):
	# Refocuses Image by Shifting Array of Images
	shifted_imgs = []
	ref_array = np.arange(289).reshape((17, 17))
	center_coord = (np.floor(ref_array.shape[0]/2.), np.floor(ref_array.shape[0]/2.))
	ref_array = np.flipud(ref_array)
	
	for x in range(ref_array.shape[0]):
		for y in range(ref_array.shape[1]):
			img = imgs[ref_array[x][y]]
			dx = int(x - center_coord[0]) * -1
			dy = int(y - center_coord[1]) * -1
			shifted_img = np.roll(img, int(dx * scale), axis=0)
			shifted_img = np.roll(shifted_img, int(dy * scale), axis=1)
			shifted_imgs.append(shifted_img)

	refocused_img = np.mean(shifted_imgs, axis=0)
	return refocused_img
