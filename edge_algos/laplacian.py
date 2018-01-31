from PIL import Image
import numpy
from scipy import signal
from scipy import misc
import math

im = Image.open('mona.jpg')
new_image = Image.new('L',im.size)
pix = im.load()
# print( im.size)

# for row in pix:
# 	for pixel in row:
# 		print( pixel)
pixels = list(im.getdata())

width, height = im.size

# for line in range(height):
# 	for cell in range(width):
# 		print(pixels[width*line + cell])

grayscale_pixels = []

for i in range(width*height):
	R = pixels[i][0]
	G = pixels[i][1]
	B = pixels[i][2]
	GS = 0.2125*R + 0.7154*G + 0.0721*B
	GS = int(GS)
	grayscale_pixels.append(GS)


new_image.putdata(grayscale_pixels)

new_image.show()

# new_image.save('new.jpg')
pixel_values = numpy.array(grayscale_pixels).reshape((height, width))

kernel = [
			[-1, -1, -1],
			[-1, 8, -1],
			[-1, -1, -1]
		]

new_pixels = []

def correlation(r,c):
	s = 0
	for i in range(3):
		for j in range(3):
			try:
				s+= kernel[i][j]*pixel_values[r-1 + i][c-1+j]
			except Exception, e:
				new_pixels.append(0)
				return
	new_pixels.append(s)


def apply_filter():
	for r in range(height):
		for c in range(width):
			correlation(r,c)


apply_filter()


new_image.putdata(new_pixels)
new_image.show()
# print(new_pixels)