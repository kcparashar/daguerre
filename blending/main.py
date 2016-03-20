# CS194-26: Computational Photography
# Project 3: Fun with Frequencies
# main.py
# Krishna Parashar

################## GLOBAL CONFIGURATIONS ############

source_dir = 'data/'
dest_dir = 'results/'
extension = '.jpg'

################## PART 0: UNSHARP ##################
'''
from unsharp import *

# Configurations
attribute = "_sharp"
img_name = "ben.jpg"
sigma = 5
alpha = 1

# Load Image from File and Normalize
print("\nStarting Part 0: Unsharp")
orig_img = plt.imread(source_dir + img_name)/255.

# Apply Unsharp Masking
sharp_img = unsharp(orig_img, sigma, alpha)

# Write Image out to File
write_image(dest_dir, img_name, sharp_img, attribute, extension)

print("Finished Part 0: Unsharp\n")


################## PART 1: Hybrid ###################

from hybrid import *

# Configurations
attribute = "_hybrid"
img_name = "dermeg.jpg"
img1_name = 'DerekPicture.jpg'
img2_name = 'nutmeg.jpg'

# Load Images from Files and Normalize
print("Starting Part 1: Hybrid")
img1 = plt.imread(source_dir + img1_name)/255.
img2 = plt.imread(source_dir + img2_name)/255.

# Align Images to Each to a Human Selected Point
img1_aligned, img2_aligned = align_images(img1, img2)

# Create the Hybrid Image
sigma1, sigma2  = 9, 4 # cutoff values for the high / low freqs
hybrid_img, high_freq, low_freq = hybrid_image(img1_aligned, img2_aligned, sigma1, sigma2)

# Compute FFT of Images
img1_fft = get_fft(color.rgb2gray(img1))
img2_fft = get_fft(color.rgb2gray(img2))
high_freq_fft = get_fft(color.rgb2gray(high_freq))
low_freq_fft = get_fft(color.rgb2gray(low_freq))
hybrid_img_fft = get_fft(color.rgb2gray(hybrid_img))

# Write Images out to File
write_gray_image(dest_dir, img_name, hybrid_img, attribute, extension)
write_gray_image(dest_dir, img_name, high_freq, "_high_freq", extension)
write_gray_image(dest_dir, img_name, low_freq, "_low_freq", extension)
write_gray_image(dest_dir, img1_name, img1_fft, "_fft", extension)
write_gray_image(dest_dir, img2_name, img2_fft, "_fft", extension)
write_gray_image(dest_dir, img_name, hybrid_img_fft, "_fft", extension)
write_gray_image(dest_dir, img_name, high_freq_fft, "_high_freq_fft", extension)
write_gray_image(dest_dir, img_name, low_freq_fft, "_low_freq_fft", extension)

print("Finished Part 1: Hybrid\n")


################## PART 2: Stack ####################

from stack import *

# Configurations
dest_dir = 'results/'
attribute = "_stacks"
img_name = "lisa.jpg"
levels = 6 

# Load Image from File and Normalize 
print("Starting Part 2: Stack")
img_hybrid = plt.imread(source_dir + img_name)/255.

# Create the Gaussian Stack
gaussian_stack = gaussian_stack(img_hybrid, levels)

# Create the Laplacian Stack
laplacian_stack = laplacian_stack(img_hybrid, levels, gaussian_stack)

# Write Stacks out to File
write_stacks(dest_dir, img_name, extension, attribute, gaussian_stack, laplacian_stack)

print("Finished Part 2: Stack\n")

'''
################## PART 3: Blend ####################

from blend import *

# Configurations
attribute = "_blend"
levels = 1
img1_side = "top"
img2_side = "bottom"
img_name = "mini.jpg"
img1_name = "mlk.jpg"
img2_name = "mlk2.jpg"
mask_name = "mlk_mask.jpg"

# Load Images and Mask from Files and Normalize
print("Starting Part 3: Blend")
img1 = plt.imread(source_dir + img1_name)/255.
img2 = plt.imread(source_dir + img2_name)/255.
# mask = np.hstack([np.ones([img1.shape[0], np.floor(img1.shape[1]/2.)]), np.zeros([img1.shape[0], np.ceil(img1.shape[1]/2.)])])
mask = plt.imread(source_dir + mask_name)/255.
mask = mask[:, :, 0]

# Compute Stacks for Each Image and Mask
img1_stack  = laplacian_stack(img1, levels, (gaussian_stack(img1, levels)))
img2_stack  = laplacian_stack(img2, levels, (gaussian_stack(img2, levels)))
mask_stack  = gaussian_stack(mask, levels)

# Apply Mask to Each Image in Image Stacks
masked_img1 = mask_imgs(img1_stack, mask_stack, img1_side)
masked_img2 = mask_imgs(img2_stack, mask_stack, img2_side)

# Sum Up Masked Images in Stacks to Create One Image
full_image = []
for img_1, img_2 in zip(masked_img1, masked_img2):
    full_image.append(img_1 + img_2)

# Multi-resolution Blend Full Image using Mask
blended_img = blend(img1, img2, img1_side, img2_side, full_image, mask, levels)
blended_img = rescale_intensity(blended_img, in_range = (0, 1), out_range=(0, 1))

import cv2
def enhance(image):
    """Modify the saturation and value of the image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)

    saturation = np.array(saturation * 1.2, dtype=np.uint16)
    saturation = np.array(np.clip(saturation, 0, 255), dtype=np.uint8)

    value = np.array(value * 1.1, dtype=np.uint16)
    value = np.array(np.clip(value, 0, 255), dtype=np.uint8)

    return cv2.cvtColor(cv2.merge((hue, saturation, value)), cv2.COLOR_HSV2BGR)

blended_img = enhance(blended_img)

# Write Image out to File
write_image(dest_dir, img_name, blended_img, attribute, extension)

print("Finished Part 3: Blend\n")
