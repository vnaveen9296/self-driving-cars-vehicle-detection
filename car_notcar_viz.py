import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob

##############################
# main
if __name__ == '__main__':
	if len(sys.argv) != 3:
		print('{} {} {}'.format(sys.argv[0], 'rootdir', 'glob_pattern'))
		sys.exit(1)

	rootdir = sys.argv[1]
	glob_pattern = sys.argv[2]

	images = glob.glob(rootdir + '\\**\\' + glob_pattern, recursive=True)
	print('Total # of *.jpeg files: ', len(images))
	cars = []
	notcars = []

	for image in images:
		if 'image' in image or 'extra' in image:
			notcars.append(image)
		else:
			cars.append(image)

	# Generate random indices for cars and notcars and display them
	fig = plt.figure(figsize=(8,6))
	for j in range(1,6):
		carind = np.random.randint(0, len(cars))
		plt.subplot(2,5,j)
		img_bgr = cv2.imread(cars[carind])
		plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
	for j in range(6,11):
		notcar_ind = np.random.randint(0, len(notcars))
		plt.subplot(2,5,j)
		img_bgr = cv2.imread(notcars[notcar_ind])
		plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
	plt.tight_layout()
	plt.show()
