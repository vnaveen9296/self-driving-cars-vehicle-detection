import cv2
import matplotlib.pyplot as plt


##############################
# main
if __name__ == '__main__':
	images = ['img_{}.jpg'.format(x) for x in range(11,15)]
	heatmaps = ['heatmap_{}.jpg'.format(x) for x in range(11,15)]
	labels = ['label_{}.jpg'.format(x) for x in range(11,15)]
	img_bboxes = ['img_with_bbox{}.jpg'.format(x) for x in range(11,15)]

	num_images = 15-11
	num_rows = num_images
	num_cols = 4
	plt_index = 1
	fig = plt.figure(figsize=(16,16))
	for j in range(num_images):
		plt.subplot(num_rows, num_cols, plt_index)
		img = cv2.imread(images[j])
		plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
		plt.title(images[j])
		plt_index += 1

		plt.subplot(num_rows, num_cols, plt_index)
		heatmap = cv2.imread(heatmaps[j])
		plt.imshow(heatmap[:,:,0], cmap='hot')
		plt.title(heatmaps[j])
		plt_index += 1

		plt.subplot(num_rows, num_cols, plt_index)
		label = cv2.imread(labels[j])
		plt.imshow(label[:,:,0], cmap='gray')
		plt.title(labels[j])
		plt_index += 1

		plt.subplot(num_rows, num_cols, plt_index)
		img = cv2.imread(img_bboxes[j])
		plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
		plt.title(img_bboxes[j])
		plt_index += 1

	plt.tight_layout()
	plt.show()