# CS194-26: Computational Photography
# Project 7: Image Warping and Mosaicing
# main.py
# Krishna Parashar

import os
import cv2
import math
import random
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import distance
from scipy.ndimage import filters, interpolation

from manual import *

# Global Configurations 
source_dir = 'data/'
extension = '.jpg' 
dest_dir = 'results/'

def write_figure(img, name, x, y):
	fig = plt.imshow(img)
	fig.set_cmap('hot')
	plot(x, y, 'r*')
	plt.axis('off')
	plt.savefig("results/" + name, bbox_inches='tight')


def harris(img):
	# Harris Corner Detector with Adaptive Non-Maximal Suppression
	# Inspired by: http://cs.colby.edu/courses/S11/cs398/download/feature.py

	# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	img = img.astype(np.float32)
	height, width = img.shape[0:2]
	
	# img_harris, corners = get_harris_corners(img) # Corners are [[y...][x...]]
	img_harris = cv2.cornerHarris(img, 2, 3, 0.04)

	img_harris[:10, :], img_harris[-10:, :] = 0, 0
	img_harris[:, :10], img_harris[:, -10:] = 0, 0

	harris_max = filters.maximum_filter(img_harris, (8,8))
	img_harris = img_harris * (img_harris == harris_max)

	indices = (np.argsort(img_harris.flatten())[::-1])[:500]
	x, y = (indices % width), (indices / width)

	return np.vstack((x, y)).T


def extract(img, corners, radius, scale):
	# Feature Descriptor Extractor
	img_filtered = filters.gaussian_filter(img, sigma=scale)
	num_corners = corners.shape[0]

	descriptors_size = (2 * radius) + 1
	descriptors = np.zeros((descriptors_size, descriptors_size, num_corners), dtype=float)

	for corner in range(num_corners):
		center_x, center_y = corners[corner, 0], corners[corner, 1]
		min_x, min_y = center_x - scale * radius, center_y - scale * radius
		max_x, max_y = center_x + scale * radius + 1, center_y + scale * radius + 1

		patch = img_filtered[min_y:max_y:scale, min_x:max_x:scale]

		patch = (patch - np.mean(patch)) / np.std(patch)
		descriptors[:, :, corner] = patch

	return descriptors


def match(img1_desc, img2_desc):
    # Feature Matcher
    tolerance = 0.4

    distances = distance.cdist(img1_desc, img2_desc)
    sorted_distances = np.argsort(distances, 1)

    best_index = sorted_distances[:, 0]
    indexes = np.arange(0, (img1_desc.shape[0]))
    
    closest_distance = distances[indexes, best_index]
    avg_second = np.mean(distances[indexes, (sorted_distances[:, 1])])

    ratio = closest_distance/avg_second

    number_matches = ratio[ratio < tolerance].shape[0]
    matches = np.zeros((number_matches, 2), dtype = np.int32)
    matches[:, 0] = indexes[ratio < tolerance]
    matches[:, 1] = best_index[ratio < tolerance]

    return matches


def ransac(matches):
	confidence = 0.95
	best_homography = np.zeros((3, 3))
	indices = np.arange(0, (matches.shape[0]))
	iterations = 0
	max_inliers = 0
	best_inlier = 0

	while (iterations < 5000):
		np.random.shuffle(indices)
		random_matches = matches[indices[:4]]
		match1, match2 = random_matches[:, :2], random_matches[:, 2:]
		
		homography = compute_homography(match2, match1)
		transformation_points = np.ones(((matches.shape[0]), 3))
		transformation_points[:, :2] = matches[:, 2:]

		transformation_matches = np.dot(homography, transformation_points.T)

		temp = np.zeros(transformation_matches.shape)
		for i in range(3):
			temp[i, :] = transformation_matches[i, :] / transformation_matches[2, :]

		transformation_matches = temp.T

		distance = np.sqrt(((matches[:, :2] - transformation_matches[:, :2])**2).sum(1))
		num_inliers = distance[distance < confidence].shape[0]

		if num_inliers > max_inliers:
			best_inlier = indices[distance < confidence]
			max_inliers = num_inliers
			best_homography = homography
		
		iterations += 1

	return best_homography


def get_homography(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img1_corners = harris(img1_gray)
    img2_corners = harris(img2_gray)

    img1_descriptors = extract(img1_gray, img1_corners, 5, 1)
    img1_descriptors = img1_descriptors.reshape(((img1_descriptors.shape[0])**2, (img1_descriptors.shape[2]))).transpose() 

    img2_descriptors = extract(img2_gray, img2_corners, 5, 1)
    img2_descriptors = img2_descriptors.reshape(((img2_descriptors.shape[0])**2, (img2_descriptors.shape[2]))).transpose() 

    matches = match(img1_descriptors, img2_descriptors)
    matches = np.hstack(((img1_corners[matches[:, 0], :2]), (img2_corners[matches[:, 1], :2])))

    homography = ransac(matches)
    return homography


##################################################################################
#                          Automatic Image Panorama                              #
##################################################################################

# Global Configurations 
source_dir = 'data/'
extension = '.jpg'
dest_dir = 'results/'

### Panorama Configurations
folder = 'london/'
img1_name = "1"
img2_name = "2"
panaroma_name = folder[:-1]
radius = 8
scale = 5

img1 = cv2.imread(source_dir + folder + img1_name + extension, -1)
img2 = cv2.imread(source_dir + folder + img2_name + extension, -1)

homography2_1 = get_homography(img2, img1)

img1_warped, shift1, img1w_shape = warp(img2, np.identity(3))

img2_warped, shift2, img2w_shape = warp(img1, homography2_1)

panorama = blend(img2_warped, img1_warped, shift2, img2w_shape)

write_img(dest_dir, panaroma_name, panorama, "_panaroma", extension)







