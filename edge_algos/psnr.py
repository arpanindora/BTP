import math
import numpy as np
from sys import argv
from scipy.misc import imread, imshow, imsave
import os

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


if __name__=="__main__":
	gt_dir = 'gt/'
	out_dir = argv[1] + '/'

	file_total = len(os.listdir(gt_dir))
	i = 1

	mean = 0

	for filename in os.listdir(gt_dir):
		if not filename.endswith('.jpg'):
			continue
		gt_im = gt_dir + filename
		out_im = out_dir + filename
		# print filename
		gt_im = imread(gt_im, mode="L")
		out_im = imread(out_im, mode="L")
		mean += psnr(gt_im,out_im)/file_total
		# print i,'of',file_total,'done.'
		i+=1
	print mean