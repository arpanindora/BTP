import math
import numpy as np
from sys import argv
from scipy.misc import imread, imshow, imsave
import os
from skimage.measure import compare_ssim as ssim


def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def ssim_value(y_true,y_pred):
	return 100*ssim(y_true, y_pred, multichannel = False)
	
if __name__=="__main__":
	gt_dir = 'gt/'
	out_dir = argv[1] + '/'

	file_total = len(os.listdir(gt_dir))
	i = 1

	mean_psnr = 0
	mean_ssim = 0
	for filename in os.listdir(gt_dir):
		if not filename.endswith('.jpg'):
			continue
		gt_im = gt_dir + filename
		out_im = out_dir + filename
		# print filename
		gt_im = imread(gt_im, mode="L")
		out_im = imread(out_im, mode="L")
		mean_psnr += psnr(gt_im,out_im)/file_total
		mean_ssim += ssim_value(gt_im,out_im)/file_total
		# print i,'of',file_total,'done.'
		i+=1
		print 'PSNR : ' +  str(mean_psnr), 'SSIM : ' +  str(mean_ssim)
	mean_psnr = str(mean_psnr)
	mean_ssim = str(mean_ssim)


	fp = open('a.txt','a')
	# fp.write(argv[1] + ': ' + str(mean_psnr) + '\n')

	s =  '=====================' + argv[1] + '=====================\n' + 'PSNR : ' + mean_psnr + '\n' + 'SSIM : ' + mean_ssim + '\n'
	fp.write(s)
	fp.close()
