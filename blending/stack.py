# CS194-26: Computational Photography
# Project 3: Fun with Frequencies
# stack.py (Part 2)
# Krishna Parashar

from unsharp import *

def gaussian_stack(img, levels):
    gaussian_stack = []
    sigma_list = [2**value for value in range(levels)]
    for sigma in sigma_list:
        gaussian_stack.append(gaussian(img, sigma))
    return gaussian_stack

def laplacian_stack(img, levels, gaussian_stack):
    laplacian_stack = []
    for item in gaussian_stack:
        laplacian_stack.append(img - item)
    return laplacian_stack

def write_stacks(dest_dir, img_name, extension, attribute, gaussian_stack, laplacian_stack):
    stack = np.concatenate([gaussian_stack, laplacian_stack])
    fig, plots = plt.subplots(nrows=2, ncols=len(gaussian_stack))
    fig.set_size_inches(20, 10)

    for (plot, img) in zip(plots.flat, stack):
        img = np.mean(rescale_intensity(img, in_range=(-0.5, 0.5), out_range=(0, 1)), axis=2)
        plot.imshow(img, cmap='gray')
        plot.axis('off')

    fig.tight_layout(h_pad=1)
    plt.savefig(dest_dir + img_name[:-4] + attribute + extension, dpi=500)
