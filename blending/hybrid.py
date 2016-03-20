# CS194-26: Computational Photography
# Project 3: Fun with Frequencies
# hybrid.py (Part 1)
# Krishna Parashar

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage import color
from align import *
from unsharp import *

def get_fft(img):
	return (np.log(np.abs(np.fft.fftshift(np.fft.fft2(img)))))

def hybrid_image(img1, img2, sigma1, sigma2):
	high_freq = laplacian(img1, sigma1)
	low_freq = gaussian(img2, sigma2)
	hybrid_img = (high_freq + low_freq)/2.
	hybrid_img = rescale_intensity(hybrid_img, in_range=(0, 1), out_range=(0, 1))
	return hybrid_img, high_freq, low_freq

def write_gray_image(dest_dir, img_name, img, attribute, extension):
    # Saves image to directory using the appropriate extension
    output_filename = dest_dir + img_name[:-4] + attribute + extension
    plt.imsave(str(output_filename), img, cmap = cm.Greys_r)
    print("Image {0} saved to {1}".format(img_name, output_filename))
