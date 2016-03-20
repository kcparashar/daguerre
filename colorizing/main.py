# CS194-26: Computational Photography
# Project 1: Colorizing the Prokudin-Gorskii photo collection.
# main.py
# Krishna Parashar

from colorize import *

# Configurations
source_dir = 'data/'
dest_dir = 'results/'
extension = '.jpg'
# img_names =   ['bridge.tif','cathedral.jpg','emir.tif',
			   # 'harvesters.tif','icon.tif','lady.tif',
			   # 'melons.tif','monastery.jpg','nativity.jpg',
			   # 'onion_church.tif','self_portrait.tif','settlers.jpg',
			   # 'three_generations.tif','tobolsk.jpg','train.tif',
			   # 'turkmen.tif','village.tif','workshop.tif']
img_names = ['arch.tif', 'dock.tif', 'patio.tif', 'sculpture.tif', 'snow.tif']
search_radius = 7

for img_name in img_names:

	# Load Image from File
	orig_img = sk.img_as_float(skio.imread(source_dir + img_name))

	# Split Image by Colors
	blue_img, green_img, red_img = map(crop, split(orig_img))

	# Align Colors
	# Naive Alignment (Uncomment to Run)	
	# displacement_green = naive_align(blue_img, green_img, search_radius)
	# aligned_green = align(green_img, displacement_green)
	# displacement_red = naive_align(blue_img, red_img, search_radius)
	# aligned_red = align(red_img, displacement_red)

	# Pyramid Alignment
	displacement_green = pyramid_align(blue_img, green_img, search_radius)
	aligned_green = align(green_img, displacement_green)
	displacement_red = pyramid_align(blue_img, red_img, search_radius)
	aligned_red = align(red_img, displacement_red)

	print("Displacing {0} (Green: {1}, Red: {2})".format((img_name[:-4]).capitalize(), displacement_green, displacement_red))
	
	# Compose into a Single Image
	color_img = stack_images(aligned_red, aligned_green, blue_img)

	# Write Image out to File
	write_image(dest_dir, img_name, color_img, extension)

