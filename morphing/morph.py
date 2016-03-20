# CS194-26: Computational Photography
# Project 5: Face Morphing
# morph.py
# Krishna Parashar

import os
import glob
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

from skimage.draw import polygon
from scipy.spatial import Delaunay
from scipy.interpolate import interp2d

def read_image(source_dir, img, extension):
	return plt.imread(source_dir + img + extension)

def get_shape(img, img_name, extension):
	shape_filename = "data/" + img_name + "_points" + extension
	if os.path.isfile(shape_filename): 
		shape = np.loadtxt(shape_filename)
		
	else: 
		img_plot = plt.imshow(img)
		prev_input, x_coor, y_coor =  (0, 0), [], []

		while True:
			x, y = plt.ginput(1)[0]
			x, y = int(x), int(y)
			if (x, y) == prev_input: break
			prev_input = (x, y)
			x_coor.append(x); y_coor.append(y);
			plt.plot(x_coor, y_coor, 'y')
			plt.draw()
		plt.close('all')

		coordinates = np.array((np.array(x_coor), np.array(y_coor)))
		shape = coordinates.T
		tlc, trc, blc, brc = [0, 0], [0, img.shape[0]], [img.shape[1], 0], [img.shape[1], img.shape[0]]
		shape = np.vstack([shape, np.array([tlc, trc, blc, brc])])
		np.savetxt(shape_filename, shape)

	return shape


def get_affine(triangle1, triangle2):
	# Calculates Transition Matrix between Two Triangles
	row1 = (np.vstack([triangle1.T, [1, 1, 1]]).T)
	row2 = (np.vstack([triangle2.T, [1, 1, 1]]).T)
	try: trans_matrix = (la.solve(row1, row2)).T
	except la.LinAlgError: return np.identity(3)
	return trans_matrix


def warp_image(img, org_shape, final_shape, triangle):
	# Warps Image shape to desired shape
	height, width, channels = img.shape

	org_points = org_shape[triangle.simplices].copy()
	final_points = final_shape[triangle.simplices].copy()
	
	trans_matrices = []
	for org_point, final_point in zip(org_points, final_points):
		trans_matrices.append(get_affine(final_point, org_point))
  
	warped_img = np.zeros(img.shape)

	for org_point, final_point, trans_matrix in zip(org_points, final_points, trans_matrices):
		rr, cc = polygon(final_point.T[1], final_point.T[0])
		mask = np.zeros(img.shape)
		mask[rr, cc] = 1
		warped_y, warped_x = np.where(mask[:, :, 0])
		warped_points = np.around(np.vstack([warped_x, warped_y, np.ones(warped_x.shape)])).astype(int)
		colors = np.around(trans_matrix.dot(warped_points)).astype(int)
		warped_img[warped_points[1], warped_points[0], :] = img[colors[1], colors[0], :]
	return warped_img


def morph(img1, img2, shape1, shape2, warp_frac):
	# Produces Warped Original Images to Average Shape
	avg_shape = np.around(((1 - warp_frac) * shape1) + (warp_frac * shape2))
	triangle = Delaunay(avg_shape)

	img1_warped = warp_image(img1, shape1, avg_shape, triangle)
	img2_warped = warp_image(img2, shape2, avg_shape, triangle)

	midway_img = (img1_warped/2) + (img2_warped/2)
	return img1_warped, img2_warped, midway_img


def write_image(dest_dir, img_name, img, attribute, extension):
	# Saves image to directory using the appropriate extension
	output_filename = dest_dir + img_name + attribute + extension
	plt.imsave(str(output_filename), img/255.)
	print("Image {0} saved to {1}".format(img_name, output_filename))


def make_gif(img1, img2, img1_name, img2_name, shape1, shape2, dest_dir, extension, warp_frac):
	# Composes Frames for a post production GIF
 	wf = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

	avg_shape = np.around(((1 - warp_frac) * shape1) + (warp_frac * shape2))
	triangle = Delaunay(avg_shape)

	wf_img1, wf_img2 = np.outer((1 - wf), shape1.T), np.outer(wf, shape2.T)

	warped_shape1 = wf_img1.reshape((1 - wf).shape[0] * shape1.T.shape[0], shape1.T.shape[1])
	warped_shape2 = wf_img2.reshape(wf.shape[0] * shape2.T.shape[0], shape2.T.shape[1])
	warped_shape = np.around(warped_shape1 + warped_shape2).astype(int)

	for counter in range(wf.shape[0]):
		final_shape = warped_shape[(2 * counter) : (2 * counter + 2)].T
		img1_warped = warp_image(img1, shape1, final_shape, triangle)
		img2_warped = warp_image(img2, shape2, final_shape, triangle)
		frame = (1. - wf[counter]) * img1_warped + wf[counter] * img2_warped
		write_image(dest_dir + "gif/", img1_name + "_" + img2_name, frame, str(counter), extension)

