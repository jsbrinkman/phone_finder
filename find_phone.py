# In this script we get one image as input and then we have to predict the location of the phone on that image.
import cv2
import pickle
import numpy as np
from utils import from_window_coord_to_norm_coord, get_centroid
import matplotlib.pyplot as plt
import sys

def sliding_window(image, stepsize):
    """ Slide a window across the image to be able to detect the phone. For every window the HOG features are computed """
    height, width = image.shape

    # We'll store coords and features in these lists
    coords = []
    features = []

    # Load SubImage class as used for training the model
    filename = 'SubImage.pickle'
    load_subimage = pickle.load(open(filename, 'rb'))

    # Get window_size as used for training the model
    window_width = load_subimage.window_width
    window_height = load_subimage.window_height

    # Slide window across image and compute for every window the HOG features
    for w1, w2 in zip(range(0, width - window_width, stepsize), range(window_width, width, stepsize)):
        for h1, h2 in zip(range(0, height - window_height, stepsize), range(window_height, height, stepsize)):
            window = image[h1:h2, w1:w2]
            features_of_window, _ = load_subimage.extract_hog_features(window)
            coords.append((w1, w2, h1, h2))
            features.append(features_of_window)

    return (coords, np.asarray(features)), load_subimage

def detect_phone(image_file, model, show_image = False):
    """ From a given image, detect the location of the phone. """
    # Get image
    image = cv2.imread(image_file)

    # Convert to gray
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Sliding window
    stepsize = 16
    windows, load_subimage = sliding_window(image_gray, stepsize)

    # Examine all the windows to see if there is a phone or not
    coordinates = []
    probability = []
    for coord, features in zip(windows[0], windows[1]):
        temp = model.decision_function(np.array([features]))
        probability.append(temp[0])
        coordinates.append(coord)

    # The window with the maximum probability contains the phone, so we take the coordinates of this window
    argmax_index = np.argmax(probability)
    predict_coord = coordinates[argmax_index]

    # Obtain image
    image_predict = image_gray[predict_coord[2]:predict_coord[3], predict_coord[0]:predict_coord[1]]

    # To be more accurate: we detect the black area in image (screen of phone) and take the centroid of this area.
    try:
        corners = load_subimage.find_contours_black_screen(image_predict)
        centroid = get_centroid(corners)
        half_window_side = False
    except:
        centroid = [0,0]
        half_window_side = True

    if show_image is True:
        plt.imshow(image_predict)
        # Show contours
        image_predict_color = image[predict_coord[2]:predict_coord[3], predict_coord[0]:predict_coord[1]]
        cv2.drawContours(image=image_predict_color, contours=corners, contourIdx=-1, color=(0, 255, 0),
                         thickness=2,
                         lineType=cv2.LINE_AA)
        plt.imshow(image_predict_color)

    # Get x,y position (image indices) in total image frame
    predict_x = predict_coord[0]+centroid[0]
    predict_y = predict_coord[2]+centroid[1]

    # Get normalized coordinates of phone
    x_predict, y_predict = from_window_coord_to_norm_coord([predict_x, predict_y], image_gray, half_window_side, load_subimage.half_window_size_norm)
    return x_predict, y_predict

if __name__ == '__main__':
    ''' Main function '''
    image_file = sys.argv[1]

    # Load model
    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    # Predict location of phone
    x_predict, y_predict = detect_phone(image_file, loaded_model, show_image=True)
    print(round(x_predict,4),round(y_predict,4))
