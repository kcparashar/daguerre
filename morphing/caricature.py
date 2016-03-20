# CS194-26: Computational Photography
# Project 5: Face Morphing
# caricature.py
# Krishna Parashar

from morph import *
from mean_face import *

def caricature(img1, img2, shape1, shape2, warp_frac, distortion):
	avg_shape = np.around(((1 - warp_frac) * shape1) + (warp_frac * shape2))
	triangle = Delaunay(avg_shape)
	difference = np.subtract(shape1, avg_shape)
	final_shape = np.add(shape1, distortion * difference)
	caricature = warp_image(img1, shape1, final_shape, triangle)
	return caricature