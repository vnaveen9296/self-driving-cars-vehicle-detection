import sys
import cv2
import numpy as np
import pickle
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label
from training_pipeline_threaded import inference_pipeline, draw_boxes, add_heat, apply_threshold, draw_labeled_bboxes
#from training_pipeline import inference_pipeline, draw_boxes, add_heat, apply_threshold, draw_labeled_bboxes
from scipy.ndimage.measurements import label

data = {}
frame_number = 0
heatmaps_hist = []
####################
def process_img(img):
	'''
	img -- input image in RGB format
	'''
	global data, frame_number
	# data is a global var (dict) already loaded and contains various params required
	svc = data['svc']
	X_scaler = data['scaler']
	orient = data['orient']
	pix_per_cell = data['pix_per_cell']
	cell_per_block = data['cell_per_block']
	spatial_size = data['spatial_size']
	hist_bins = data['hist_bins']
	color_space = data['color_space']
	hog_channel = data['hog_channel']
	use_spatial_bin = data['use_spatial_bin']
	use_color_hist = data['use_color_hist']
	use_hog_features = data['use_hog_features']

	# Step1: call inference pipeline to get hot windows
	# Note: inference_pipeline() function expects BGR style image
	img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	hot_windows = inference_pipeline(img_bgr, svc, X_scaler, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel,
						use_spatial_bin, use_color_hist, use_hog_features)
	
	window_img = draw_boxes(img, hot_windows, color=(0,0,255), thick=6)
	
	# Step2: Heatmaps - create heatmap, add heat and apply thresholding
	heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
	# add heat to each bbox in list
	heatmap = add_heat(heatmap, hot_windows)
	# a history of heatmaps is maintained in heatmaps_hist variable. Push the current heatmap to the end of heatmaps_hist
	if frame_number < 10:
		heatmaps_hist.append(heatmap)
	else:
		heatmaps_hist[:-1] = heatmaps_hist[1:]
		heatmaps_hist[-1] = heatmap
	# get cumulative_heatmap from heatmaps_hist
	cumulative_heatmap = np.sum(heatmaps_hist, axis=0)
	# apply threshold to remove false positives
	cumulative_heatmap = apply_threshold(cumulative_heatmap, 15)
	labels = label(cumulative_heatmap)
	# visualize the heatmap
	cumulative_heatmap = np.clip(cumulative_heatmap, 0, 255)

	# draw labeled bboxes
	draw_img_rgb = draw_labeled_bboxes(img, labels, color=(255,255,0), thick=3)

	if frame_number >= 11 and frame_number <= 16:
		output_image_file = 'img_{}.jpg'.format(frame_number)
		output_image_file2 = 'img_with_bbox{}.jpg'.format(frame_number)
		output_heatmap_file = 'heatmap_{}.jpg'.format(frame_number)
		output_label_file = 'label_{}.jpg'.format(frame_number)
		cv2.imwrite(output_image_file, cv2.cvtColor(window_img, cv2.COLOR_RGB2BGR))
		cv2.imwrite(output_heatmap_file, cumulative_heatmap)
		cv2.imwrite(output_label_file, labels[0])
		cv2.imwrite(output_image_file2, cv2.cvtColor(draw_img_rgb, cv2.COLOR_RGB2BGR))

	# insert text
	text = 'frame num ({}), number of cars ({})'.format(frame_number, labels[1])
	cv2.putText(draw_img_rgb, text, (200,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
	#cv2.putText(window_img, text, (200,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
	frame_number += 1
	#return window_img
	#return np.hstack((window_img, draw_img_rgb))
	return draw_img_rgb

#################################
# main
if __name__ == '__main__':
	#global data
	if len(sys.argv) != 4:
		print('Usage: {} {} {} {}'.format(sys.argv[0], 'input_video_file', 'output_video_file', 'svc_pickle_file'))
		sys.exit(1)
	input_videofile = sys.argv[1]
	output_videofile = sys.argv[2]
	pickle_file = sys.argv[3]

	# load parameters from pickled file
	data = pickle.load(open(pickle_file, 'rb'))
	clip1 = VideoFileClip(input_videofile)
	outclip = clip1.fl_image(process_img)
	outclip.write_videofile(output_videofile, audio=False)