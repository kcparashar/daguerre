# CS194-26: Computational Photography
# Project 4: Seam Carving
# seamcarver.py
# Krishna Parashar

import sys
import scipy as sp
import numpy as np
import skimage as sk
import skimage.io as skio
import IPython

from skimage import color
from skimage import filters
from scipy.ndimage import sobel
from skimage.exposure import rescale_intensity


def find_energy(img):
    # Calculates importance of image using Gradient Magnitude from Sobel filter. 

    # img = color.rgb2gray(img) # Values waaay to small
    # Color Values from http://scikit-image.org/docs/dev/api/skimage.color.html
    R = 0.2125 * img[:, :, 0]
    G = 0.7154 * img[:, :, 1]
    B = 0.0721 * img[:, :, 2]
    img = np.add(np.add(R, G), B)

    # grad_mag = magnitude(dx + dy)
    dx, dy = sobel(img, 0), sobel(img, 1)
    magnitude = np.hypot(dx, dy)
    grad_mag = magnitude * (255.0 / np.max(magnitude))

    return grad_mag

def find_min_energies(raw_energies):
    # Uses Dynamic Programming to propagate energy paths
    height, width = raw_energies.shape
    energies = np.empty((height, width))
    energies[0][:] = raw_energies[0][:]
    
    for i in xrange(1, height):
        for j in xrange(0, width):
            if (j - 1) < 0:
                energies[i, j] = raw_energies[i, j] + min(energies[i - 1, j], energies[i - 1, j + 1])
            elif (j + 1) >= width:
                energies[i, j] = raw_energies[i, j] + min(energies[i - 1, j - 1], energies[i - 1, j])
            else:
                energies[i, j] = raw_energies[i, j] + min(energies[i - 1, j - 1], energies[i - 1, j], energies[i - 1, j + 1])

    return energies
    
def find_seam(energies):
    # Locates minimum cost seam to remove for give image
    height, width = energies.shape
    seam = np.zeros((height, 1))
    inf = float("Inf")

    for i in xrange(height - 1, -1, -1):
        if i == (height - 1): 
            j = np.where(energies[i][:] == min(energies[i, :]))[0][0]
        
        else:
            if seam[i + 1, 0] == 0: 
                parents = [inf, energies[i,seam[i + 1, 0]], energies[i, seam[i + 1, 0] + 1]]
            elif seam[i + 1, 0] == width - 1: 
                parents = [energies[i, seam[i + 1, 0] - 1], energies[i, seam[i + 1, 0]], inf]
            else: 
                parents = [energies[i, seam[i + 1, 0] - 1], energies[i, seam[i + 1, 0]], energies[i, seam[i + 1, 0] + 1]]
           
            position = parents.index(min(parents)) - 1
            j = seam[i + 1, 0] + position

        seam[i, 0] = j

    return seam

def remove_seam(img, seam):
	# Removes seam value passed in to image
	height, width, channels = img.shape
	carved_img = np.zeros((height, width - 1, channels))

	for k in xrange(0, channels):
	   for i in xrange(0, height):
	       if seam[i, 0] == 0:
	           carved_img[i, :, k] = img[i, 1 : width, k]
	       elif seam[i, 0] == width - 1:
	           carved_img[i, :, k] = img[i, 0 : width - 1, k]
	       else:
				left_of_seam, right_of_seam = img[i , 0 : seam[i, 0], k], img[i, seam[i, 0] + 1 : width, k]
				carved_img[i, :, k] = np.append(left_of_seam, right_of_seam)

	return carved_img
