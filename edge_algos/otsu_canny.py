import numpy as np
from scipy.ndimage.filters import convolve, gaussian_filter
from scipy.misc import imread, imshow, imsave
import sys
import matplotlib.pyplot as plt
import os
import glob
import math
from skimage.measure import compare_ssim as ssim

np.set_printoptions(threshold=np.nan)

def find_mean(hist,bins):

	mean = 0.0
	summed = 0.0
	for i in range(hist.shape[0]):
		summed += hist[i]
		mid_point = (bins[i] + bins[i+1])/2

		mean += hist[i]*mid_point

	if summed == 0:
		return 0
	return mean/summed

def find_std(hist,bins):

	mean = find_mean(hist,bins)
	total = 0.0
	std_dev = 0.0
	for i in range(hist.shape[0]):
		total += hist[i]
		std_dev += np.power(hist[i] - mean,2.0)

	if total == 0:
		return 0

	return np.power(std_dev/total,0.5)
 
def CannyEdgeDetector(im, blur = 1, highThreshold = 91, lowThreshold = 31,H = 1, show_hist = False, bins = 512, K = 1, L = 0):
	im = np.array(im, dtype=float) #Convert to float to prevent clipping values
 
	#Gaussian blur to reduce noise
	# im2 = gaussian_filter(im, blur)
 
	#Use sobel filters to get horizontal and vertical gradients
	im2h = convolve(im,[[-1,0,1],[-2,0,2],[-1,0,1]])
	im2v = convolve(im,[[1,2,1],[0,0,0],[-1,-2,-1]])
 
	#Get gradient and direction
	W = np.exp(- (np.power(np.power(im2h, 2.0) + np.power(im2v, 2.0), 0.5)/(np.power(H, 2.0)*2)))


	im3 = np.zeros((im.shape[0],im.shape[1]))

	# print W.shape
	for i in range(im.shape[0]):
		for j in range(im.shape[1]):

			if i == 0 or i == im.shape[0]-1 or j == 0 or j == im.shape[1]-1:
				continue

			D = 0
			V = 0
			for k in range(-1,2):
				for l in range(-1,2):
					D += W[i+k,j+l]

					V += im[i+k,j+l]*W[i+k,j+l]
			
			im3[i,j] = V/D

	im4h = convolve(im3,[[-1,0,1],[-2,0,2],[-1,0,1]])
	im4v = convolve(im3,[[1,2,1],[0,0,0],[-1,-2,-1]])

	grad = np.power(np.power(im4h, 2.0) + np.power(im4v, 2.0), 0.5)

	theta = np.arctan2(im4v, im4h)
	thetaQ = (np.round(theta * (5.0 / np.pi)) + 5) % 5 #Quantize direction
 
	#Non-maximum suppression
	gradSup = grad.copy()
	for r in range(im.shape[0]):
		for c in range(im.shape[1]):
			#Suppress pixels at the image edge
			if r == 0 or r == im.shape[0]-1 or c == 0 or c == im.shape[1] - 1:
				gradSup[r, c] = 0
				continue
			tq = thetaQ[r, c] % 4
 
			if tq == 0: #0 is E-W (horizontal)
				if grad[r, c] <= grad[r, c-1] or grad[r, c] <= grad[r, c+1]:
					gradSup[r, c] = 0
			if tq == 1: #1 is NE-SW
				if grad[r, c] <= grad[r-1, c+1] or grad[r, c] <= grad[r+1, c-1]:
					gradSup[r, c] = 0
			if tq == 2: #2 is N-S (vertical)
				if grad[r, c] <= grad[r-1, c] or grad[r, c] <= grad[r+1, c]:
					gradSup[r, c] = 0
			if tq == 3: #3 is NW-SE
				if grad[r, c] <= grad[r-1, c-1] or grad[r, c] <= grad[r+1, c+1]:
					gradSup[r, c] = 0


	flattened_grad = np.reshape(grad,(grad.shape[0])*(grad.shape[1]))

	mean_grad = np.mean(flattened_grad)

	hist, bin_edges = np.histogram(flattened_grad,bins=bins)

	# print hist,bin_edges

	total = (grad.shape[0])*(grad.shape[1])

	otsu_thresh = 0.0
	sum_n = 0.0
	max_between_variance = -1.0


	for i in range(hist.shape[0]):


		sum_n =  sum_n + hist[i]
		
		PA = (sum_n/total)
		PB = (total - sum_n)/total

		mu_A = find_mean(hist[:i+1],bin_edges[:i+2])
		mu_B = find_mean(hist[i+1:],bin_edges[i:])

		# print PA,PB,mu_A,mu_B

		between_variance = PA*np.power(mu_A - mean_grad, 2.0) + PB*np.power(mu_B - mean_grad,2.0)

		# print between_variance


		if max_between_variance < between_variance:
			max_between_variance = between_variance
			otsu_thresh = i


	# print otsu_thresh

	mu_weak = find_mean(hist[:otsu_thresh+1],bin_edges[:otsu_thresh+2])
	mu_strong = find_mean(hist[otsu_thresh+1:],bin_edges[otsu_thresh:])
	
	std_weak = find_std(hist[:otsu_thresh+1],bin_edges[:otsu_thresh+2])
	std_strong = find_std(hist[otsu_thresh+1:],bin_edges[otsu_thresh:])
	
	# print 'mu_weak:',mu_weak,'mu_strong:',mu_strong
	# print 'std_weak:',std_weak,'std_strong:',std_strong
	#Double threshold

	highThreshold = mu_weak + K*std_weak
	lowThreshold = mu_weak + L*std_weak

	# print highThreshold,lowThreshold

	if show_hist == True:
		plt.hist(flattened_grad,bins=bins)  # arguments are passed to np.histogram
		plt.title("Histogram with %d bins" % (bins))

		plt.axvline(mean_grad, color='b', linestyle='dashed', linewidth=2)
		plt.axvline(highThreshold, color='r', linestyle='dashed', linewidth=2)
		plt.axvline(lowThreshold, color='g', linestyle='dashed', linewidth=2)

		plt.show()

	strongEdges = (gradSup > highThreshold)
 
	#Strong has value 2, weak has value 1
	thresholdedEdges = np.array(strongEdges, dtype=np.uint8) + (gradSup > lowThreshold)
 	
	#Tracing edges with hysteresis	
	#Find weak edge pixels near strong edge pixels
	finalEdges = strongEdges.copy()
	currentPixels = []
	for r in range(1, im.shape[0]-1):
		for c in range(1, im.shape[1]-1):	
			if thresholdedEdges[r, c] != 1:
				continue #Not a weak pixel
 
			#Get 3x3 patch	
			localPatch = thresholdedEdges[r-1:r+2,c-1:c+2]
			patchMax = localPatch.max()
			if patchMax == 2:
				currentPixels.append((r, c))
				finalEdges[r, c] = 1

	#Extend strong edges based on current pixels
	while len(currentPixels) > 0:
		newPix = []
		for r, c in currentPixels:
			for dr in range(-1, 2):
				for dc in range(-1, 2):
					if dr == 0 and dc == 0: continue
					r2 = r+dr
					c2 = c+dc
					if thresholdedEdges[r2, c2] == 1 and finalEdges[r2, c2] == 0:
						#Copy this weak pixel to final result
						newPix.append((r2, c2))
						finalEdges[r2, c2] = 1
		currentPixels = newPix
 
	return finalEdges
 
def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def ssim_value(y_true,y_pred):
	return 100*ssim(y_true, y_pred, multichannel = False)

def eval(argv,K,L):
	K = str(K)
	L = str(L)

	gt_dir = 'test_gt/'
	out_dir = argv

	file_total = len(os.listdir(path))

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
	
	print 'PSNR : ' +  str(mean_psnr), 'SSIM : ' +  str(mean_ssim), "K : " + K, "L : " + L

	fp = open('b.txt','a')
	# fp.write(argv[1] + ': ' + str(mean_psnr) + '\n')

	s =  '=====================' + argv + '=====================' + '\n' + 'PSNR : ' + str(mean_psnr) + '\n' + 'SSIM : ' + str(mean_ssim) + '\n' + "K : " + K + '\n' + "L : " + L + '\n'
	fp.write(s)
	fp.close()
	return mean_psnr,mean_ssim

if __name__=="__main__":
	path = 'test_data/'

	path_save = 'otsu_canny/'

	if not os.path.exists(path_save):
		os.makedirs(path_save)

	file_total = len(os.listdir(path))


	min_psnr = 100
	max_ssim = 0
	
	best_K_psnr = 0
	best_L_psnr = 0

	best_K_ssim = 0
	best_L_ssim = 0

	for i in range(0,3):
		for j in range(i + 1,3):
			count = 1
			for filename in os.listdir(path):
				if not filename.endswith('.jpg'):
					continue
				input_im = path + filename
				print filename
				im = imread(input_im, mode="L") #Open image, convert to greyscale
				K = j/10.0
				L = i/10.0
				finalEdges = CannyEdgeDetector(im,show_hist=False,K = K , L = L)

				imsave(path_save + filename, finalEdges)
				print count,'of',file_total,'done.'
				count+=1
			temppsnr, tempssim = eval(path_save,K,L)

			if temppsnr < min_psnr:
				best_K_psnr = K
				best_L_ssim = L

			if tempssim > max_ssim:
				best_K_ssim = K
				best_L_ssim = L

	fp = open('b.txt','a')
	s = '********************************\n' + 'Minimum PSNR : ' + str(min_psnr) + " K : " + str(best_K_psnr) + " L : " + str(best_L_psnr) + '\n' + 'Maximum SSIM : ' + str(max_ssim)  + " K : " + str(best_K_ssim) + " L : " + str(best_L_ssim) + '\n'
	fp.write(s)
	fp.close()

	count = 1
	for filename in os.listdir(path):
		if not filename.endswith('.jpg'):
			continue
		input_im = path + filename
		print filename
		im = imread(input_im, mode="L") #Open image, convert to greyscale

		finalEdges = CannyEdgeDetector(im,show_hist=False)
		imsave(path_save + filename,finalEdges)
		print count,'of',file_total,'done.'
		count+=1


	# input_im = sys.argv[1]
	# im = imread(input_im, mode="L") #Open image, convert to greyscale
	# finalEdges = CannyEdgeDetector(im,bins = 256)
	# imshow(finalEdges)
