import os
from sys import argv

i = 0
for file in os.listdir('data'):
	if file not in os.listdir('gt'):
		os.remove('data/' + file)
		print file,i
		i+=1


for file in os.listdir('gt'):
	if file not in os.listdir('data'):
		os.remove('gt/' + file)
		print file,i
		i+=1


