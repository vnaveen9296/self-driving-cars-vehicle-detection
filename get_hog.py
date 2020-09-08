import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from skimage.feature import hog
import ipdb

##############################
# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
	if vis == True:
		features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), 
				cells_per_block=(cell_per_block,cell_per_block), visualise=True, feature_vector=feature_vec, transform_sqrt=True)
		return features, hog_image
	else:
		features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), 
				cells_per_block=(cell_per_block,cell_per_block), visualise=False, feature_vector=feature_vec, transform_sqrt=True)
		return features


##############################
def convert_to_colorspace(img_bgr, desired_color_space):
	img = img_bgr.copy()
	if desired_color_space == 'RGB':
		img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
	elif desired_color_space == 'HSV':
		car_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
	elif desired_color_space == 'HLS':
		car_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
	elif desired_color_space == 'YCrCb':
		car_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
	return img

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

	# Generate a random index to look at a car image and not car image
	carind = np.random.randint(0, len(cars))
	notcar_ind = np.random.randint(0, len(notcars))

	# Define HOG parameters
	#orient = 9
	#pix_per_cell = 8
	#cell_per_block = 2
	parameters = [(9,8,2), (11,8,2),(9,8,4),(11,8,4),(9,16,2), (11,16,2)]	# list of tuples where each tuple is of the form (orient, pix_per_cell, cell_per_block)

	# Read the image
	car_img_bgr = cv2.imread(cars[carind])
	notcar_img_bgr = cv2.imread(notcars[notcar_ind])
	# conver to gray
	car_gray = cv2.cvtColor(car_img_bgr, cv2.COLOR_BGR2GRAY)
	notcar_gray = cv2.cvtColor(notcar_img_bgr, cv2.COLOR_BGR2GRAY)
	# convert to different color space if needed
	car_img = convert_to_colorspace(car_img_bgr, 'YCrCb')
	notcar_img = convert_to_colorspace(car_img_bgr, 'YCrCb')
	
	fig = plt.figure(figsize=(12,8))
	#fig = plt.figure()
	num_rows = 3	#len(parameters) + 1
	num_cols = len(parameters)
	# plot car and notcar in the 1st row
	plt.subplot(num_rows, num_cols, 1)
	plt.imshow(car_gray, cmap='gray')
	plt.title('car')
	plt.subplot(num_rows, num_cols, 2)
	plt.imshow(notcar_gray, cmap='gray')
	plt.title('notcar')

	plt_index = num_cols + 1
	for j,params in enumerate(parameters):
		orient, pix_per_cell, cell_per_block = params		
		# Call HOG function vis=True to see an image output
		features, car_hog_image_gray = get_hog_features(car_gray, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
		plt.subplot(num_rows, num_cols, plt_index+j)
		plt.imshow(car_hog_image_gray, cmap='gray')
		plt.title('{}'.format(params))

	plt_index += num_cols
	for j,params in enumerate(parameters):
		orient, pix_per_cell, cell_per_block = params		
		# Call HOG function vis=True to see an image output
		features, notcar_hog_image_gray = get_hog_features(notcar_gray, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)

		# plot car HOG and notcar HOG
		plt.subplot(num_rows, num_cols, plt_index+j)
		plt.imshow(notcar_hog_image_gray, cmap='gray')
		plt.title('{}'.format(params))
	plt.tight_layout()
	plt.show()