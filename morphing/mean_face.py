# CS194-26: Computational Photography
# Project 5: Face Morphing
# meanface.py
# Krishna Parashar


from morph import *

def mean_face(imgs, shapes):
	# Calculates the Mean Face of A Set of Images
	x_points, y_points = shapes[::2], shapes[1::2]
	avg_x_points, avg_y_points = np.mean(x_points, axis=0), np.mean(y_points, axis=0)
	avg_shape = np.around(np.array([avg_x_points, avg_y_points]).T).astype(int)
	triangle = Delaunay(avg_shape)
	warped_imgs = []
	
	for img, x_point, y_point in zip(imgs, x_points, y_points):
		org_shape = np.array([x_point, y_point]).T
		final_shape = avg_shape
		warped_imgs.append(warp_image(img, org_shape, final_shape, triangle))

	mean_face_img = np.sum(np.array(warped_imgs)/len(warped_imgs), axis=0)
	return mean_face_img  
