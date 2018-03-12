import numpy as np
from scipy.ndimage.filters import convolve, gaussian_filter
from scipy.misc import imread, imshow, imsave
import sys
import matplotlib.pyplot as plt
import os
import glob

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
 
def CannyEdgeDetector(im, blur = 1, highThreshold = 91, lowThreshold = 31,H=1, K = 1):
	im = np.array(im, dtype=float) #Convert to float to prevent clipping values
 
	#Gaussian blur to reduce noise
	# im2 = gaussian_filter(im, blur)
 
	#Use sobel filters to get horizontal and vertical gradients
	im2h = convolve(im,[[-1,0,1],[-2,0,2],[-1,0,1]])
	im2v = convolve(im,[[1,2,1],[0,0,0],[-1,-2,-1]])
 
	#Get gradient and direction
	W = np.exp(- (np.power(np.power(im2h, 2.0) + np.power(im2v, 2.0), 0.5)/(np.power(H, 2.0)*2)))


	im3 = np.zeros((im.shape[0],im.shape[1]))

	print W.shape
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
	std_dev = np.std(flattened_grad)


	total = (grad.shape[0])*(grad.shape[1])

	highThreshold = mean_grad + K*mean_grad
	lowThreshold = highThreshold/2

	# print highThreshold,lowThreshold

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
 
if __name__=="__main__":
	path = 'data/'
	path_save = 'old_canny/'

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
		finalEdges = CannyEdgeDetector(im,K=1.2)
		imsave(path_save + filename,finalEdges)
		# imshow(finalEdges)
		print i,'of',file_total,'done.'
		i+=1

	# input_im = sys.argv[1]
	# im = imread(input_im, mode="L") #Open image, convert to greyscale
	# finalEdges = CannyEdgeDetector(im,K=1.2)
	# imshow(finalEdges)
