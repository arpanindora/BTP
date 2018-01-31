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
			[0.0438 , 0.0982, 0.108, 0.0982,0.0438],
			[0.0982, 0, -0.242, 0, 0.0982],
			[0.108, -0.242, -0.7979, -0.242, 0.108],
			[0.0982, 0, -0.242, 0, 0.0982],
			[0.0438 , 0.0982, 0.108, 0.0982,0.0438]
		]

log_pixels = []

def correlation(r,c):
	s = 0
	for i in range(5):
		for j in range(5):
			try:
				s+= kernel[i][j]*pixel_values[r-2 + i][c-2+j]
			except Exception, e:
				pass
	log_pixels.append(s)


def apply_filter():
	for r in range(height):
		for c in range(width):
			correlation(r,c)


apply_filter()



# new_pixels = numpy.array(pixel_values).reshape((height*width))

new_image.putdata(log_pixels)
new_image.show()
# print(new_pixels)