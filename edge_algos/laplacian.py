import numpy as np
from scipy.ndimage.filters import convolve, gaussian_filter
from scipy.misc import imread, imshow, imsave
import sys
import matplotlib.pyplot as plt
import os
import glob

np.set_printoptions(threshold=np.nan)

def Laplacian(im, blur = 1):
	im = np.array(im, dtype=float)
	im2 = convolve(im,[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
 	
	return im2
 
if __name__=="__main__":
	path = 'data/'

	path_save = 'Laplacian/'

	if not os.path.exists(path_save):
		os.makedirs(path_save)

	file_total = len(os.listdir(path))
	i = 1
	for filename in os.listdir(path):
		if not filename.endswith('.jpg'):
			continue
		input_im = path + filename
		print filename
		im = imread(input_im, mode="L") #Open image, convert to greyscale
		finalEdges = Laplacian(im)
		imsave(path_save + filename,finalEdges)
		print i,'of',file_total,'done.'
		i+=1

	# input_im = sys.argv[1]
	# im = imread(input_im, mode="L") #Open image, convert to greyscale
	# finalEdges = Laplacian(im)
	# imshow(finalEdges)
