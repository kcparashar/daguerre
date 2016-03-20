# CS194-26: Computational Photography
# Project 4: Seam Carving
# main.py
# Krishna Parashar

from seamcarver import *

# Configurations
source_dir = 'data/'
dest_dir = ''
extension = '.jpg'
img_name = "monkey.jpg"
new_width = 700 # pixels (750px normally)
new_height = 475 # pixels (500px normally) 

# Load Image from File
img = skio.imread(source_dir + img_name)

# Calculate Number of Seams to Remove
height, width, channels = img.shape
diff_width = width - int(new_width)
diff_height = height - int(new_height)

# Remove Vertical Seams
for i in range (0, diff_width):
    print "Vertical Seams: Removed", i, "seams of", diff_width
    raw_energies = find_energy(img)
    min_energy = find_min_energies(raw_energies)
    seam = find_seam(min_energy)
    img = remove_seam(img, seam)

# Remove Horizontal Seams
img = img.transpose(1, 0, 2)
for i in range (0, diff_height):
    print "Horizontal Seams: Removed", i, "seams of", diff_height
    raw_energies = find_energy(img)
    min_energy = find_min_energies(raw_energies)
    seam = find_seam(min_energy)
    img = remove_seam(img, seam)
img = img.transpose(1, 0, 2)/255.

# Write Image out to File
skio.imsave(img_name[:-4] + str(new_width) + "_" + str(new_height) + str(extension), img)
