import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
import ipdb
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import time
from skimage.feature import hog
import pickle
import threading
import math

####################
def draw_boxes(img, bboxes, color=(0,0,255), thick=6):
    '''Draw boxes on image.
    img -- input image
    bboxes - list of boxes. The list is a tuple of pairs where each pair represents a rectangle
    '''
    draw_img = img.copy()
    for box in bboxes:
        cv2.rectangle(draw_img, box[0], box[1], color, thick)

    return draw_img

####################
def draw_labeled_bboxes(img, labels, color=(0,0,255), thick=6):
    '''Draw labeled boxes on image.
    img -- input image
    labels - labels found (cars found)
    '''
    for car_number in range(1, labels[1]+1):
        # find pixels with each car_number label value
        nz = (labels[0] == car_number).nonzero()
        # get x and y values of these pixels
        nzy = np.array(nz[0])
        nzx = np.array(nz[1])
        # define a bounding box based on min/max of x and y
        bbox = ((np.min(nzx), np.min(nzy)), (np.max(nzx),np.max(nzy)))
        # draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], color, thick)
    return img

####################
def add_heat(heatmap, bbox_list):
    '''Add heat to a map for a given list of bounding boxes.
    heatmap -- the map to which we want to add heat
    bbox_list -- bounding boxes
    '''
    # iterate through the list of boxes
    for bbox in bbox_list:
        # Add += 1 for all pixels inside each bbox
        heatmap[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] += 1

    # return the updated heatmap
    return heatmap

####################
def apply_threshold(heatmap, threshold):
    '''Threshold the heatmap (for rejecting areas affected by false positives).
    heatmap -- input heatmap
    threshold -- threshold value to apply
    '''
    # zero out pixels below threshold
    heatmap[heatmap <= threshold] = 0
    return heatmap

####################
# Define a function that takes an image, a color space and a size and returns a feature vector
def bin_spatial(img, size=(32,32)):
    '''Spatial binning (used as features).
    img -- input image
    size -- represent the (re)size that needs to be applied to img
    Return value -- feature vector
    '''
    features = cv2.resize(img, size).ravel()
    return features

####################
# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    '''Compute Color histograms (used as features)
    img -- input image
    nbins -- number of histogram bins
    bins_range - total range
    Return value -- feature vector
    '''
    ch1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    ch2_hist = np.histogram(img[:,:,1], nbins, range=bins_range)
    ch3_hist = np.histogram(img[:,:,2], nbins,range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.hstack((ch1_hist[0], ch2_hist[0], ch3_hist[0]))
    return hist_features

####################
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    '''Get HOG features.
    img -- input image
    orient -- number of orientation bins
    pix_per_cell -- size of a cell (in number of pixels)
    cell_per_block -- number of cells in each block
    vis -- flag to determine if HOG visualization image needs to be returned or not
    feature_vec -- flag to determine if the features need to be returned as a feature vector

    Return value -- HOG feature vector and HOG visualization image
    '''
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), 
                cells_per_block=(cell_per_block,cell_per_block), visualise=True, feature_vector=feature_vec, transform_sqrt=True)
        return features, hog_image
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), 
                cells_per_block=(cell_per_block,cell_per_block), visualise=False, feature_vector=feature_vec, transform_sqrt=True)
        return features

####################
def convert_color_space(imagefile, desired_color_space):
    '''Read image using cv2.imread() and return it in desired colorspace.
    imagefile -- input image file
    desired_color_space -- desired color space

    Return value -- image in the desired color space
    '''
    img_bgr = cv2.imread(imagefile)
    if desired_color_space == 'RGB':
        outimg = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    elif desired_color_space == 'HSV':
        outimg = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    elif desired_color_space == 'HLS':
        outimg = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
    elif desired_color_space == 'YUV':
        outimg = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
    elif desired_color_space == 'LUV':
        outimg = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LUV)
    elif desired_color_space == 'YCrCb':
        outimg = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    return outimg


####################
def extract_features(imagefiles, cspace='RGB', spatial_size=(32,32), hist_bins=32, hist_range=(0,256),
                        orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, use_spatial_bin=True,
                        use_color_hist=True, use_hog_features=True):
    '''Extract features from a list of imagefiles.'''

    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for imagefile in imagefiles:
        # Step1: Apply color conversion
        feature_image = convert_color_space(imagefile, color_space)

        # Step2: use spatial binning
        if use_spatial_bin:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
        # Step3: use color histograms
        if use_color_hist:
            hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)

        # Step4: use HOG features
        # call get_hog_features with vis=False, feature_vec=True
        if use_hog_features:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], orient, pix_per_cell, cell_per_block, vis=False,
                                            feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Step5: features from all - create one long vector of features
        stacked = []
        if use_spatial_bin:
            stacked.append(spatial_features)
        if use_color_hist:
            stacked.append(hist_features)
        if use_hog_features:
            stacked.append(hog_features)
        stacked = np.hstack(stacked)
        # extracting features from one imagefile is complete - append it to the list that we will return
        features.append(stacked)
    return features


####################
# Define a function to extract features from a single image (or single image window)
# This function is very similar to extract_features() but just for a single image rather than list of images
def single_img_features(img, desired_color_space, spatial_size=(32,32), hist_bins=32, orient=9, pix_per_cell=8, cell_per_block=2,
                            hog_channel=0, use_spatial_bin=True, use_color_hist=True, use_hog_features=True):
    '''img -- input image as read by cv2.imread() - expected in BGR format'''
    img_features = []
    # Step1: Apply color conversion
    if desired_color_space == 'RGB':
        feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif desired_color_space == 'HSV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif desired_color_space == 'HLS':
        feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    elif desired_color_space == 'YUV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    elif desired_color_space == 'LUV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    elif desired_color_space == 'YCrCb':
        feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # Step2: Use spatial binning
    if use_spatial_bin:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        img_features.append(spatial_features)
    # Step3: Use color histograms
    if use_color_hist:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        img_features.append(hist_features)

    # Step4: Use HOG features
    # call get_hog_features with vis=False, feature_vec=True
    if use_hog_features:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], orient, pix_per_cell, cell_per_block, vis=False,
                                        feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        img_features.append(hog_features)
    # stack all features i.e. create one long vector of features
    img_features = np.hstack(img_features)
    return img_features
    

####################
# Define a function that takes an input image and list of windows (output from slide_windows())
# and searches the windows for cars
def search_windows(img, windows, clf, scaler, color_space, spatial_size=(32,32), hist_bins=32, hist_range=(0,256), orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0, use_spatial_bin=True, use_color_hist=True, use_hog_features=True):
    '''Search (all possible) windows for cars.
    img -- input image read by cv2.imread() - expected in BGR format
    '''
    # create an empty list to receive positive detection windows
    positive_windows = []
    t = time.time()
    for window in windows:
        # extract the test window from the original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64,64))
        # extract features from test_img using single_img_features()
        features = single_img_features(test_img, desired_color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins, orient=orient,
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                            use_spatial_bin=use_spatial_bin, use_color_hist=use_color_hist, use_hog_features=use_hog_features)
        # scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # predict using the classifier
        prediction = clf.predict(test_features)
        # if positive prediction (i.e. car detected), then save the window
        if prediction == 1:
            positive_windows.append(window)
    t2 = time.time()
    #print('Time took for search_windows(): ', round(t2-t, 2), 'seconds and t = {}, t2 = {}'.format(t, t2))
    # return windows for positive detections
    return positive_windows


####################
def create_sliding_windows(img, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(64,64), xy_overlap=(0.5,0.5)):
    '''Create a list of possible sliding windows.
    img -- input image
    x_start_stop -- start and stop positions for x
    y_start_stop -- start and stop positions in y direction
    xy_window -- sliding window size
    xy_overlap -- overlap in x and y direction while creating sliding windows
    '''
    # if x, y start or stop positions are None, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]

    # compute number of windows in xy
    # Initialize a list to append window positions to
    window_list = []
    xstep = int(xy_window[0] * (1-xy_overlap[0]))
    ystep = int(xy_window[1] * (1-xy_overlap[1]))
    for ypos in range(y_start_stop[0], y_start_stop[1]-xy_window[1]+1, ystep):
        for xpos in range(x_start_stop[0], x_start_stop[1]-xy_window[0]+1, xstep):
            window_list.append(((xpos, ypos),(xpos+xy_window[0], ypos+xy_window[1])))

    #print('Number of windows = ', len(window_list))
    return window_list


####################
def get_data(rootdir, glob_pattern):
    '''Returns lists of images - one for cars and another for notcars.
    rootdir -- the directory where to look recursively for image files
    glob_pattern -- file pattern to match (ex: *.png or *.jpeg)
    ''' 
    images = glob.glob(rootdir + '\\**\\' + glob_pattern, recursive=True)
    print('Total # of Image files: ', len(images))
    cars = []
    notcars = []

    for image in images:
        #if 'image' in image or 'extra' in image:
        if 'non-vehicles' in image:
            notcars.append(image)
        else:
            cars.append(image)
    return (cars, notcars)

####################
# training pipeline
def training_pipeline(rootdir, glob_pattern, spatial_size, num_hist_bins, color_space, use_spatial_bin, use_color_hist, use_hog_features):
    '''Training Pipeline for the classifier.

    After the classifier is trained, all the parameters are saved to a pickle file. Generated pickled filename is named as svc_{color_space}.pkl
    '''
    # Step1: get image data
    cars, notcars = get_data(rootdir, glob_pattern)


    # tweak these parameters and see how the results change
    #color_space = 'HSV'    # 'YCrCb'   # 'RGB'
    orient = 11
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = 'ALL' # can be 0,1,2 or ALL
    spatial_size = (spatial,spatial)
    num_hist_bins = num_hist_bins
    y_start_stop = [None, None]

    # Step2: Extract features
    car_features = extract_features(cars, cspace=color_space, spatial_size=spatial_size, hist_bins=num_hist_bins, hist_range=(0,256), orient=orient,
                            hog_channel=hog_channel, use_spatial_bin=use_spatial_bin, use_color_hist=use_color_hist, 
                            use_hog_features=use_hog_features)
    notcar_features = extract_features(notcars, cspace=color_space, spatial_size=spatial_size, hist_bins=num_hist_bins, hist_range=(0,256), orient=orient,
                            hog_channel=hog_channel, use_spatial_bin=use_spatial_bin, use_color_hist=use_color_hist, 
                            use_hog_features=use_hog_features)

    # Step3: Normalize features
    # create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    #car_ind = np.random.randint(0, len(cars))

    # Step4: Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Step5: split the data into randomized training and test sets
    rand_state = np.random.randint(0,100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Feature vector length: ', len(X_train[0]))

    # Step5: Train a classifier -- Using a linear SVC
    svc = LinearSVC()
    # check the training time for SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print('Took',round(t2-t, 2), 'Seconds to train SVC...')
    
    # Step6: Check the score of the svc (test accuracy)
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    # Step7: save svc and other params to a pickle
    output_pickle_file = 'svc_{}.pkl'.format(color_space)
    with open(output_pickle_file, 'wb') as f:
        data = { 'svc': svc,
                'scaler': X_scaler,
                'orient': orient,
                'pix_per_cell': pix_per_cell,
                'cell_per_block': cell_per_block,
                'spatial_size' : spatial_size,
                'hist_bins': num_hist_bins,
                'color_space': color_space,
                'hog_channel': hog_channel,
                'use_spatial_bin': use_spatial_bin,
                'use_color_hist': use_color_hist,
                'use_hog_features': use_hog_features
                }
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    print(output_pickle_file, 'is written')
    return output_pickle_file

####################
# Inference pipeline. Takes an image and returns hot windows (i.e. list of rectangle where cars are detected)
def inference_pipeline(img, svc, X_scaler, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel,
                        use_spatial_bin, use_color_hist, use_hog_features):
    '''img -- input image as read by cv2.imread() - expected in BGR format'''
    y_start_stop = [int(img.shape[0]/2), img.shape[0]]
    hot_windows = []
    t = time.time()
    for sz, overlap in zip([80, 96, 128], [0.5, 0.75, 0.75]):
        possible_windows = create_sliding_windows(img, x_start_stop=[None, None], y_start_stop=y_start_stop, xy_window=(sz,sz), xy_overlap=(overlap,overlap))
        #ipdb.set_trace()
        # img is in BGR format - as expected by search_windows() function
        pos_wins = search_windows(img, possible_windows, svc, X_scaler, color_space=color_space, spatial_size=spatial_size,
                            hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                            hog_channel=hog_channel, use_spatial_bin=use_spatial_bin, use_color_hist=use_color_hist, 
                            use_hog_features=use_hog_features)
        hot_windows.extend(pos_wins)
    t2 = time.time()
    #print('Time took for creating and search windows: ', round(t2-t, 2), 'Seconds')
    return hot_windows


#####################################
# main
if __name__ == '__main__':
    '''
    Example Usage:
    For training:
        python classify.py vehicles_non_vehicles *.png 32 32 YCrCb 1 1 1 test_images\test1.jpeg 1
    For testing an image using already generated pickle file:
        python classify.py vehicles_non_vehicles *.png 32 32 YCrCb 1 1 1 test_images\test1.jpeg 0
    '''
    if len(sys.argv) != 11:
        print('Usage: {} {} {} {} {} {} {} {} {} {} {}'.format(sys.argv[0], 'rootdir', 'glob_pattern', 'spatial_bin_size', 'num_hist_bins',
                                'color_space', 'use_spatial_bin', 'use_color_hist', 'use_hog_features', 'test_image', 'do_training'))
        sys.exit(1)

    rootdir = sys.argv[1]
    glob_pattern = sys.argv[2]
    spatial = int(sys.argv[3])
    spatial_size = (spatial, spatial)
    num_hist_bins = int(sys.argv[4])
    color_space = sys.argv[5]
    use_spatial_bin = int(sys.argv[6])
    use_color_hist = int(sys.argv[7])
    use_hog_features = int(sys.argv[8])
    test_imagefile = sys.argv[9]
    do_training = int(sys.argv[10])
    #print(rootdir, spatial, num_hist_bins, use_spatial_bin, use_color_hist, use_hog_features)

    if do_training == 1:
        pickle_file = training_pipeline(rootdir, glob_pattern, spatial_size, num_hist_bins, color_space, use_spatial_bin, use_color_hist, use_hog_features)
    else:
        # use pickle file based on color space
        pickle_file = 'svc_{}.pkl'.format(color_space)

    # load from pickle file
    data = pickle.load(open(pickle_file, 'rb'))
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
    print('Parameters are loaded from pickle file {}'.format(pickle_file))

    # Step8: classifier is ready. Apply it on a test image for detecting cars
    img = cv2.imread(test_imagefile)
    hot_windows = inference_pipeline(img, svc, X_scaler, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel,
                        use_spatial_bin, use_color_hist, use_hog_features)
    # visualize: for drawing purpose, get an rgb image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    window_img = draw_boxes(img_rgb, hot_windows, color=(200, 0, 0), thick=3)
    plt.imshow(window_img)
    plt.show()

