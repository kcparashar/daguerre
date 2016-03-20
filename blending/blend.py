# CS194-26: Computational Photography
# Project 3: Fun with Frequencies
# stack.py (Part 3)
# Krishna Parashar

from stack import *

def mask_img(img, mask, side):
    if (side == "left") or (side == "top"): 
        multiplier = mask
    elif (side == "right") or (side == "bottom"): 
        multiplier = 1 - mask

    channels = []
    channels.append(img[:, :, 0] * multiplier) #np.pad(multiplier, 3, mode="median")
    channels.append(img[:, :, 1] * multiplier)
    channels.append(img[:, :, 2] * multiplier)
    
    return np.dstack(channels)

def mask_imgs(img_stack, mask_stack, side):
    masked_stack = []
    for img, mask in zip(img_stack, mask_stack):
        masked_stack.append(mask_img(img, mask, side))
    return masked_stack

def blend(img1, img2, img1_side, img2_side, img_stack, mask, levels):
    sigma = 2**(levels)
    gaussian_mask = gaussian(mask, sigma)
    img1_low_freq = mask_img(gaussian(img1, sigma), gaussian_mask, img1_side)
    img2_low_freq = mask_img(gaussian(img2, sigma), gaussian_mask, img2_side)
    return sum(img_stack, sum([img1_low_freq, img2_low_freq]))
