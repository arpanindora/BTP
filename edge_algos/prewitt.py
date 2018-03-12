import numpy as np
from scipy.ndimage.filters import convolve, gaussian_filter
from scipy.misc import imread, imshow, imsave
import sys
import matplotlib.pyplot as plt
import os
import glob

def PrewittEdgeDetector(im):
	im = np.array(im, dtype=float)

	im2h = convolve(im,[[-1,0,1],[-1,0,1],[-1,0,1]])
	im2v = convolve(im,[[1,1,1],[0,0,0],[-1,-1,-1]])

	grad = np.power(np.power(im2h, 2.0) + np.power(im2v, 2.0), 0.5)

	return grad

if __name__=="__main__":
	path = 'data/'

	path_save = 'prewitt/'

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
		finalEdges = PrewittEdgeDetector(im)
		imsave(path_save + filename,finalEdges)
		print i,'of',file_total,'done.'
		i+=1

	# input_im = sys.argv[1]
	# im = imread(input_im, mode="L") #Open image, convert to greyscale
	# finalEdges = SobelEdgeDetector(im)
	# imshow(finalEdges)
