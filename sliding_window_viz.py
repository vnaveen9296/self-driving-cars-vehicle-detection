import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import ipdb

def draw_boxes(img, bboxes, color=(0,0,255), thick=6):
    draw_img = img.copy()
    for box in bboxes:
        cv2.rectangle(draw_img, box[0], box[1], color, thick)

    return draw_img

def draw_boxes2(img, bboxes, color=(0,0,255), color2=(255,0,0), thick=6):
    draw_img = img.copy()
    colors = [color] * len(bboxes)
    colors[1::2] = [color2] * len(colors[1::2])
    for c, box in zip(colors, bboxes):
        cv2.rectangle(draw_img, box[0], box[1], c, thick)

    return draw_img

# Define a function that takes an image, 
# start and stop positions in both x and y,
# window size (x and y dimensions)
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(64,64), xy_overlap=(0.5,0.5)):
    # if x, y start or stop positions are None, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]

    assert xy_overlap[0] < 1.0 and xy_overlap[1] < 1.0, 'Overlap must be between [0.0,1)'
    # compute number of windows in xy
    # Initialize a list to append window positions to
    window_list = []
    xstep = int(xy_window[0] * (1-xy_overlap[0]))
    ystep = int(xy_window[1] * (1-xy_overlap[1]))
    for ypos in range(y_start_stop[0], y_start_stop[1]-xy_window[1]+1, ystep):
        for xpos in range(x_start_stop[0], x_start_stop[1]-xy_window[0]+1, xstep):
            window_list.append(((xpos, ypos),(xpos+xy_window[0], ypos+xy_window[1])))
    return window_list


##############################
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('{} {}'.format(sys.argv[0], 'image_file'))
        sys.exit(1)

    img = cv2.imread(sys.argv[1])

    y_start_stop = [int(img.shape[0]/2), img.shape[0]]
    #y_start_stop = [None, None]

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    windows = []
    for sz, overlap in zip([80, 96, 128], [0.5, 0.75, 0.75]):
        wins = slide_window(img, x_start_stop=[None, None], y_start_stop=y_start_stop, xy_window=(sz,sz), xy_overlap=(overlap, overlap))
        windows.extend(wins)
    #window_img = draw_boxes2(img_rgb, windows, color=(0,0,255), color2=(255,0,0), thick=2)
    window_img = draw_boxes(img_rgb, windows, color=(100,100,255), thick=2)

    print('{},{},{}'.format(sz, overlap, len(windows)))
    
    plt.figure()
    plt.imshow(window_img)
    plt.title('sliding windows')
    plt.show()
    