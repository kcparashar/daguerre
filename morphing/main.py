# CS194-26: Computational Photography
# Project 5: Face Morphing
# main.py
# Krishna Parashar


from morph import *
from mean_face import *
from caricature import *


# Configurations
source_dir = 'data/'
dest_dir = 'results/'
extension = '.jpg'
attribute = '_warped'
img1_name = "obama"
img2_name = "mccain"
warp_frac = 0.5
distortion = 1.1

print ("EXPECTED RUN TIME: 60 seconds \n")
# Load Images from File and Ensure Equal Dimensions
print ("1/3: Morphing Images")
img1 = read_image(source_dir, img1_name, extension)
img2 = read_image(source_dir, img2_name, extension)
assert img1.shape == img2.shape

# Calculate Shapes from Selected Points and Ensure Equal Number of Points
shape1 = get_shape(img1, img1_name, ".txt")
shape2 = get_shape(img2, img2_name, ".txt")
assert shape1.shape == shape2.shape

# Create Image Morphs
img1_warped, img2_warped, midway_img = morph(img1, img2, shape1, shape2, warp_frac)

# # Write Images to Files
# write_image(dest_dir, img1_name, img1_warped, attribute, extension)
# write_image(dest_dir, img2_name, img2_warped, attribute, extension)
# write_image(dest_dir, img1_name + "_" + img2_name, midway_img, "", extension)

# # Makes Transitional GIF Frames of the Morph
# make_gif(img1, img2, img1_name, img2_name, shape1, shape2, dest_dir, extension, warp_frac)
# print ("Finished Morphing Images\n")


# # Makes Mean Image of Danish Populations
# print ("2/3: Creating a Mean Image of Danish Populations")
# danish_imgs_src = "data/imm_face_db/*-2*.jpg"
# danish_shapes_src = "data/imm_face_db/danish_shape.txt"
# face_img_names = (glob.glob(danish_imgs_src))[:2] + (glob.glob(danish_imgs_src))[5:]
# danish_shapes = np.loadtxt(danish_shapes_src)
# danish_imgs = []
# for img_name in face_img_names:
# 	danish_imgs.append(read_image("", img_name, ""))
# mean_face_img = mean_face(danish_imgs, danish_shapes)
# write_image(dest_dir, "danish", mean_face_img, "_average", extension)
# print ("Finished Creating a Mean Image of Danish Populations\n")


# Makes Caricature 
print ("3/3: Caricaturing my Face")

img1 = read_image(source_dir, "me", extension)
img2 = read_image(source_dir, "bollywood", extension)
assert img1.shape == img2.shape

shape1 = get_shape(img1, "me", ".txt")
shape2 = get_shape(img2, "bollywood", ".txt")
assert shape1.shape == shape2.shape

caricature = caricature(img1, img2, shape1, shape2, warp_frac, distortion)
write_image(dest_dir, "caricature", caricature, "_me", extension)
print ("Finished Caricaturing my Face")
