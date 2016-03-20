# CS194-26: Computational Photography
# Project 6: Lightfield Camera
# main.py
# Krishna Parashar

from aperture import *
from refocus import *

# Configurations 
source_dir = 'data/'
folder = "chess/" # Configured to work with Chess Dataset
dest_dir = 'results/'
extension = '.png'
dof_range = (-3, 1) # OPTIMAL RANGE [-3, 1]
step = 1
aperture_radius = 2 # RANGE [0, 8]

# Loads images in from directory
imgs = read_images(source_dir, folder, extension)

# Produces all the depth refocused images in the set range
dofs = np.arange(dof_range[0], dof_range[1] + step, step)
for scale in dofs:
	print ("Refocusing image with scale of {0}".format(scale))
	refocused_img = refocus(imgs, scale)
	write_images(dest_dir + "dof/" + folder, folder[:-1], refocused_img, "_focus", str(scale), extension)

# Produces all the aperture adjusted images up until desired radius
for aperture in range(0, aperture_radius):
	adjusted_img = adjust_aperture(imgs, aperture)
	write_images(dest_dir + "aperture/" + folder, folder[:-1], adjusted_img, "_aperture", str(aperture), extension)
