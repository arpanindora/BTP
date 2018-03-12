import numpy as np
from scipy.ndimage.filters import convolve, gaussian_laplace
from scipy.misc import imread, imshow, imsave
import sys
import os

np.set_printoptions(threshold=np.nan)

def LOG(im, blur = 1):
	im = np.array(im, dtype=float)
	im2 = gaussian_laplace(im,blur)
	return im2
 
if __name__=="__main__":
	path = 'data/'

	path_save = 'LOG/'

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
		finalEdges = LOG(im)
		imsave(path_save + filename,finalEdges)
		print i,'of',file_total,'done.'
		i+=1

	# input_im = sys.argv[1]
	# im = imread(input_im, mode="L") #Open image, convert to greyscale
	# finalEdges = LOG(im)
	# imshow(finalEdges)
