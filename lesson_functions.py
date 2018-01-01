import numpy as np
import cv2
from skimage.feature import hog
from scipy.ndimage.measurements import label


def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


# define a function to return HOG feature and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, visualise=vis,
                                  feature_vector=feature_vec)
        return features, hog_image
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, visualise=vis,
                                  feature_vector=feature_vec)
        return features


# define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    return features


# define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features


# define a function to extract features from a list of images
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32), hist_bins=32, orient=9, pix_per_cell=8,
                     cell_per_block=2, hog_channel=0, use_spatial_feature=True, use_hist_feature=True,
                     use_hog_feature=True):
    features = []
    for file in imgs:
        file_features = []
        image = cv2.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        else:
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if use_spatial_feature:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)

        if use_hist_feature:
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)

        if use_hog_feature:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(
                        get_hog_features(feature_image[:, :, channel], orient, pix_per_cell, cell_per_block, vis=False,
                                         feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient, pix_per_cell, cell_per_block,
                                                vis=False,
                                                feature_vec=True)

            # append hog features to the features list
            file_features.append(hog_features)
        features.append((np.concatenate(file_features)))
    return features


# extract features from one image
def extract_img_features(image, color_space='RGB', spatial_size=(32, 32), hist_bins=32, orient=9, pix_per_cell=8,
                         cell_per_block=2, hog_channel=0, use_spatial_feature=True, use_hist_feature=True,
                         use_hog_feature=True):
    features = []

    # apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    else:
        feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if use_spatial_feature:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        features.append(spatial_features)

    if use_hist_feature:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        features.append(hist_features)

    if use_hog_feature:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(
                    get_hog_features(feature_image[:, :, channel], orient, pix_per_cell, cell_per_block, vis=False,
                                     feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient, pix_per_cell, cell_per_block,
                                            vis=False,
                                            feature_vec=True)

        # append hog features to the features list
        features.append(hog_features)

    return np.concatenate(features).reshape(1, -1)


# define a function that takes an image, start and stop positions in both x and y, window size (x and y dimensions), and overlay fraction (for both x and y)
def slide_window(img_size, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(64, 64),
                 xy_overlap=(0.5, 0.5)):
    # if x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img_size[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop = img_size[0]

    # compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]

    # compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))

    # compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)

    window_list = []
    # loop through finding x and y window positions
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            window_list.append(((startx, starty), (endx, endy)))

    # return the list of windows
    return window_list


# define a function to get all the sliding windows according to the settings
def get_sliding_windows(image_size, settings):
    output = []
    for window_size, y_limits, overlap in settings:
        windows = slide_window(img_size=image_size, x_start_stop=[None, None], y_start_stop=y_limits,
                               xy_window=window_size, xy_overlap=overlap)
        output.extend(windows)
    return output


# define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    imcopy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1],
                      (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)), thick)
    return imcopy


def add_heat(heatmap, bbox_list):
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap


def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap


def draw_labeled_boxes(img, labels):
    img_copy = np.copy(img)
    for car_number in range(1, labels[1] + 1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        cv2.rectangle(img_copy, bbox[0], bbox[1], (0, 0, 255), 6)
    return img_copy


#find cars in a single image
def find_cars(img, sliding_windows, model):
    object_windows = []
    for window in sliding_windows:
        # get the patch
        minx = window[0][0]
        maxx = window[1][0]
        miny = window[0][1]
        maxy = window[1][1]
        patch = img[miny:maxy, minx:maxx, :]

        # resize to (64,64)
        resized_patch = cv2.resize(patch, (64, 64))

        # calculate features
        patch_feature = extract_img_features(resized_patch, color_space=model['param_color_space'],
                                             spatial_size=model['param_spatial_size'],
                                             hist_bins=model['param_hist_bins'], orient=model['param_orient'],
                                             pix_per_cell=model['param_pix_per_cell'],
                                             cell_per_block=model['param_cell_per_block'],
                                             hog_channel=model['param_hog_channel'],
                                             use_spatial_feature=model['param_use_spatial_feature'],
                                             use_hist_feature=model['param_use_hist_feature'],
                                             use_hog_feature=model['param_use_hog_feature'])

        scaled_X = model['X_scaler'].transform(patch_feature)
        # classify the patch
        class_result = model['svc'].predict(scaled_X)
        if class_result[0] == 1:
            object_windows.append(window)

    # create heat map
    heat = np.zeros_like(img[:, :, 0].astype(np.float))
    heat = add_heat(heat, object_windows)
    heat = apply_threshold(heat, 1)  # threshold for single image;
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)

    # draw the boxest and labels
    img_sliding_windows = draw_boxes(img, sliding_windows, thick=2)  # image with all the sliding windows
    img_object_windows = draw_boxes(img, object_windows, thick=2)  # image with all the object windows
    img_labels = draw_labeled_boxes(img, labels)  # image with vehicles
    return object_windows, labels, img_sliding_windows, img_object_windows, heatmap, img_labels
