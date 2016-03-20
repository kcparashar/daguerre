# CS194-26: Computational Photography
# Project 7: Image Warping and Mosaicing
# main.py
# Krishna Parashar

import os
import cv2
import scipy
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp2d


# Global Configurations 
source_dir = 'data/'
extension = '.jpg'
dest_dir = 'results/'


def write_img(dest_dir, img_name, img, attribute, extension):
	# Saves image to directory using the appropriate extension
	output_filename = dest_dir + img_name + attribute + extension
	cv2.imwrite(str(output_filename), img)
	print("Image {0} saved to {1}".format(img_name, output_filename))


def set_correspondences(img1, img2, img1_name, img2_name, folder, extension='.txt'):
	def select_points(img, img_name, folder, extension):
		shape_filename = "data/" + folder + img_name + "_points" + extension

		if os.path.isfile(shape_filename): 
			shape = np.loadtxt(shape_filename)
	        
		else: 
			print ("Please select correspondence points for the image.")
			img_plot = plt.imshow(img)
			plt.suptitle('Please select points')
			points, x_coor, y_coor =  (0, 0), [], []
			plt.xlim(0, img.shape[1])
			plt.ylim(img.shape[0], 0)

			while True:
			    x, y = plt.ginput(n=1, timeout=0)[0]
			    x, y = int(x), int(y)
			    if points and (x, y) == points: break;
			    points = (x, y)
			    x_coor.append(x); y_coor.append(y);
			    plt.plot(x_coor, y_coor,  'bs')
			    plt.draw()
			plt.close('all')

			coordinates = np.array((np.array(x_coor), np.array(y_coor)))
			shape = coordinates.T
			np.savetxt(shape_filename, shape)

		return shape

	img1_points = select_points(img1, img1_name, folder, extension)
	img2_points = select_points(img2, img2_name, folder, extension)
	return img1_points, img2_points


def get_homography(source_points, target_points):
	source_xs = source_points.flatten()[0::2]
	source_ys = source_points.flatten()[1::2]

	target_xs = target_points.flatten()[0::2]
	target_ys = target_points.flatten()[1::2]

	assert (len(source_xs) == len(source_ys))
	assert (len(target_xs) == len(target_ys))
	assert (len(source_xs) == len(target_ys))

	A = []
	for i in range(0, len(source_xs)):
		A.append([source_xs[i], source_ys[i], 1, 0, 0, 0, ((-source_xs[i]) * target_xs[i]), ((-source_ys[i]) * target_xs[i])])
		A.append([0, 0, 0, source_xs[i], source_ys[i], 1, ((-source_xs[i]) * target_ys[i]), ((-source_ys[i]) * target_ys[i])])
	A = np.array(A)

	b = target_points.flatten().T

	a, b, c, d, e, f, g, h = np.linalg.lstsq(A, b)[0]
	homography = np.array([[a, b, c], [d, e, f], [g, h, 1]])

	return homography

def warp(source_img, homography):
	# source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2BGRA)
	height, width = source_img.shape[:2] # x, y

	### Finding the shape of the Warped Image
	max_width, max_height = int(-100000), int(-100000)
	min_width, min_height = int(100000), int(100000)
	orig_corners = np.float32([[0, 0], [height - 1, 0], [0, width - 1], [height - 1, width - 1]])

	warped_corners = []
	for corner in orig_corners:
		x = corner[0];
		y = corner[1];

		warped_corner = homography.dot([y, x, 1])

		warped_corner[0] = int(warped_corner[0]/warped_corner[2])
		warped_corner[1] = int(warped_corner[1]/warped_corner[2])
		warped_corner[2] = int(1)

		cur_width = warped_corner[0]  # j
		cur_height = warped_corner[1] # i

		if cur_height < min_height: min_height = cur_height
		if cur_height > max_height: max_height = cur_height

		if cur_width < min_width: min_width = cur_width
		if cur_width > max_width: max_width = cur_width

		warped_corners.append(warped_corner.tolist())

	warped_height = max_height - min_height + 1
	warped_width = max_width - min_width + 1
	warped_img_shape = (warped_height, warped_width, 3) 
	shift_height = -min_height; shift_width = -min_width;
	warped_img = np.zeros(warped_img_shape)

	for i in range(0, int(warped_height)):
		for j in range(0, int(warped_width)):
			source_point = np.linalg.inv(homography).dot([j - shift_width, i - shift_height, 1])
			source_point[0] = int(source_point[0]/source_point[2]) # j
			source_point[1] = int(source_point[1]/source_point[2]) # i
			source_point[2] = int(1)
			if (0 <= source_point[1] <= height - 1) and (0 <= source_point[0] <= width - 1): 
				warped_img[i, j, :] = source_img[source_point[1], source_point[0], :]

	return warped_img, (shift_height, shift_width), warped_img_shape


def blend(img, center_img, shift, imgw_shape):
	shift_height, shift_width = shift
	shift_height, shift_width = int(shift_height), int(shift_width)

	imgw_height, imgw_width = imgw_shape[:2]

	ci_height, ci_width = center_img.shape[:2]

	mosaic_height = ci_height + shift_height
	mosaic_width = ci_width + shift_width

	if imgw_height > mosaic_height: mosaic_height = imgw_height
	if imgw_width > mosaic_width: mosaic_width = imgw_width

	mosaic = np.zeros((mosaic_height, mosaic_width, 3))

	for i in range(0, int(ci_height)):
		for j in range(0, int(ci_width)):
			mosaic[i + shift_height, j + shift_width, :] = center_img[i, j, :]

	for i in range(0, int(imgw_height)):
		for j in range(0, int(imgw_width)):
			if (int(img[i, j, 0]) != 0 and int(img[i, j, 0]) != 0 and int(img[i, j, 2]) != 0):
				mosaic[i, j, :] = img[i, j, :]

	return mosaic




'''
##################################################################################
#                           Image Rectifications                                 #
##################################################################################

### Rectification Configurations
folder = 'rectify/'
# rectified_img_name = "holstee"
rectified_img_name = "graph"

### Load Image
print("Beginning Rectification of image {0}".format(rectified_img_name + extension))
img = cv2.imread(source_dir + folder + rectified_img_name + extension, cv2.IMREAD_COLOR)

### Setup Corresspondences Points
img_points, rectification_points = set_correspondences(img, np.zeros(img.shape), rectified_img_name, rectified_img_name + "_rectified", folder)

### Calculate Homography
homography = get_homography(img_points, rectification_points)

### Recify Image Using Homography
rectified_img, shift_rect, img_rect_shape = warp(img, homography)

## Write Image Out
write_img(dest_dir, rectified_img_name, rectified_img, "_rectified", extension)


'''
##################################################################################
#                              Image Panorama                                    #
##################################################################################

### Panorama Configurations
folder = 'campanile/'
img1_name = "1"
img2_name = "2"
img3_name = "3"
panaroma_name = folder[:-1]

### Load Images
print("Constructing Panorama of {0}".format(panaroma_name))

img1 = cv2.imread(source_dir + folder + img1_name + extension, cv2.IMREAD_COLOR)
img2 = cv2.imread(source_dir + folder + img2_name + extension, cv2.IMREAD_COLOR)
img3 = cv2.imread(source_dir + folder + img3_name + extension, cv2.IMREAD_COLOR)

### Setup Corresspondences Points
img1_points, img1_2_points = set_correspondences(img1, img2, img1_name, img1_name + "_" + img2_name, folder)
# img3_points, img3_2_points = set_correspondences(img3, img2, img3_name, img3_name + "_" + img2_name, folder)

### Calculate Homographies
homography1_2 = get_homography(img1_points, img1_2_points)
# homography3_2 = get_homography(img3_points, img3_2_points)

### Warp Each Image to `img2` Using Homography
img1_warped, shift1, img1w_shape = warp(img1, homography1_2)
# img3_warped, shift3, img3w_shape = warp(img3, homography3_2)

### Create Mosaic Using Warped Images and Center Image (`img2`)
img1_2 = blend(img1_warped, img2, shift1, img1w_shape)
# img2_3 = blend(img3_warped, img2, shift3, img3w_shape)


# img3_points, img3_2_points = set_correspondences(img3, img2, img3_name, img3_name + "_" + img2_name, folder)
# homography3_2 = get_homography(img3_points, img3_2_points)
# homography3_2 = get_homography(img3_points, img3_2_points)


# panorama = blend(img3_warped, img1_2, shift3, img3w_shape)

### Write Images Out
write_img(dest_dir, panaroma_name, img1_warped, "_1", extension) ####
write_img(dest_dir, panaroma_name, img2, "_2", extension)        ####
# write_img(dest_dir, panaroma_name, img2_3, "_3", extension) ####
write_img(dest_dir, panaroma_name, img1_2, "_panaroma", extension) ####

# write_img(dest_dir, panaroma_name, panorama, "_panaroma", extension)


