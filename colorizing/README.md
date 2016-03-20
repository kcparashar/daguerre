# CS194-26: Computational Photography
# Project 1: Colorizing the Prokudin-Gorskii photo collection.
# README.txt
# Krishna Parashar

Dependencies: numpy, sci-kit image, scipy

## main.py
This file contains the code that calls the appropriate functions for each image. It lists a set of configurations: source directory for images, destination directory for images, desired extensions, image names, and search radius. It then loops through all the images provided (as a list of image names, and call the appropriate functions from colorize.py)

## colorize.py
This file contains all the functions that actually perform the alignments on the images. There is a function to split, crop, find the displacement vector, align, align it naively, image pyramid align, stack image, and write the image to file.