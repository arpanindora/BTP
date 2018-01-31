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

new_pixels = []

for i in range(width*height):
	R = pixels[i][0]
	G = pixels[i][1]
	B = pixels[i][2]
	GS = 0.2125*R + 0.7154*G + 0.0721*B
	GS = int(GS)
	new_pixels.append(GS)


new_image.putdata(new_pixels)

new_image.show()

# new_image.save('new.jpg')
pixel_values = numpy.array(new_pixels).reshape((height, width))

x_kernel = [
			[1, 0, -1],
			[1, 0, -1],
			[1, 0, -1]
		]
y_kernel = [
			[1, 1, 1],
			[0, 0, 0],
			[-1, -1, -1]
		]
x_pixelvalues = []
y_pixelvalues = []

def apply_filter():
	for r in range(height):
		for c in range(width):
			correlation_x(r,c)
			correlation_y(r,c)

def correlation_x(r,c):
	s = 0
	for i in range(3):
		for j in range(3):
			try:
				s+= x_kernel[i][j]*pixel_values[r-1 + i][c-1+j]
			except Exception, e:
				x_pixelvalues.append(0)
				return
	x_pixelvalues.append(s)
			

def correlation_y(r,c):
	s = 0
	for i in range(3):
		for j in range(3):
			try:
				s+= y_kernel[i][j]*pixel_values[r-1 + i][c-1+j]
			except Exception, e:
				y_pixelvalues.append(255)
				return
	y_pixelvalues.append(s)

apply_filter()


for i in range(len(new_pixels)):
	# try:
	s = x_pixelvalues[i]*x_pixelvalues[i] + y_pixelvalues[i]*y_pixelvalues[i]
	new_pixels[i] = math.sqrt(int(s))
	# except Exception, e:
		# pass
	# print math.sqrt(s)
# new_pixels = numpy.array(pixel_values).reshape(height*width)
# print new_pixels
new_image.putdata(new_pixels)
new_image.show()
# print(new_pixels)