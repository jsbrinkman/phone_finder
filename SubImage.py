from skimage.feature import hog
import cv2
import numpy as np

class SubImage(object):
    """ Sub-image class, containing all functions that should only be applied to the sub-image"""
    def __init__(self, window_size_norm):
        self.window_size_norm = window_size_norm
        self.half_window_size_norm = self.window_size_norm/2

    def create_window(self, image, x_norm, y_norm):
        """ Create a smaller extract of the image (=window) with center x_norm,y_norm"""

        height, width = image.shape

        # Start corner of the window
        x_top_left = max(0, (x_norm - self.half_window_size_norm) * width)
        y_top_left = max(0, (y_norm - self.half_window_size_norm) * height)

        # Compute end corner (assure size of window is correct)
        max_x_bottom_right = width
        max_y_bottom_right = height

        # End corner of the window (this corrects for the bounds of the image)
        x_bottom_right = x_top_left + self.window_size_norm * width
        if x_bottom_right > max_x_bottom_right:
            x_bottom_right = max_x_bottom_right
            x_top_left = x_bottom_right - self.window_size_norm * width
        y_bottom_right = y_top_left + self.window_size_norm * height
        if y_bottom_right > max_y_bottom_right:
            y_bottom_right = max_y_bottom_right
            y_top_left = y_bottom_right - self.window_size_norm * height

        # Extract of image
        window = image[int(y_top_left):int(y_bottom_right), int(x_top_left):int(x_bottom_right)]

        self.window_width = window.shape[1]
        self.window_height = window.shape[0]

        return window

    def extract_hog_features(self, window):
        """ Extract the HOG features from the window """

        hog_features, hog_image = hog(window, orientations=9, pixels_per_cell=(16, 16),
                                  cells_per_block=(2,2), visualize=True, multichannel=False)

        return hog_features, hog_image

    def find_contours_black_screen(self, image_predict):
        """ Create corner points of black screen of phone for more accurate phone location"""
        # apply binary thresholding
        ret, thresh = cv2.threshold(image_predict, 50, 255, cv2.THRESH_BINARY_INV)

        # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

        # Make list of polygons with its corners
        corners_approx_total = []
        area_total = []
        for cnt in contours:
            epsilon = 0.1 * cv2.arcLength(cnt, True)
            corners_approx_total.append(cv2.approxPolyDP(cnt, epsilon, True))
            area_total.append(cv2.contourArea(cnt))

        # Black screen is the maximum area detected
        max_area = np.argmax(area_total)
        return corners_approx_total[max_area]
