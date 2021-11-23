import random
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pickle
import sys
import pandas as pd
import os
import cv2
from SubImage import SubImage
from utils import get_images, normalized_phone_coordinates


def get_training_data(folder, folder_with_images, num_of_background_images=20, window_size_norm=0.2):
    """ Create sub-images with phone and no phone and extract the HOG features. """

    data_frame = pd.read_table(folder + '/labels.txt', sep=' ', names=['image', 'x', 'y'])
    hog_features = []
    labels = [] # 1 = positive (phone), 0 = negative (no phone)
    subimage = SubImage(window_size_norm)
    half_window_size_norm = window_size_norm/2

    for image_path in folder_with_images:
        image = cv2.imread(image_path)
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image_file_name = os.path.basename(os.path.normpath(image_path))
        x_norm, y_norm = normalized_phone_coordinates(data_frame, image_file_name)

        # Create sub-images with phone (positive)
        pos_window = subimage.create_window(image_gray, x_norm, y_norm)

        # Extract HOG features
        hog_feature, _ = subimage.extract_hog_features(pos_window)
        hog_features.append(hog_feature)
        labels.append(1)

        # Create sub-images without phone (negative (N)), where N >> P
        for i in range(num_of_background_images):
            x_background = random.random()
            y_background = random.random()
            while x_background > x_norm - half_window_size_norm and x_background < x_norm + half_window_size_norm:
                x_background = random.random()
            while y_background > y_norm - half_window_size_norm and y_background < y_norm + half_window_size_norm:
                y_background = random.random()
            neg_window = subimage.create_window(image_gray, x_background, y_background)

            # Extract hog features
            hog_feature, _ = subimage.extract_hog_features(neg_window)
            hog_features.append(hog_feature)
            labels.append(0)

    # Save the class
    filename = 'SubImage.pickle'
    pickle.dump(subimage, open(filename, 'wb'))

    return hog_features, labels

def train_model(train_data, train_label):
    """ Train SVM classifier"""
    clf = svm.SVC()
    clf.fit(train_data, train_label)
    print("Model trained")
    return clf

def test_model(model, test_data, test_label):
    """ This is the accuracy of predicting if the image is a phone or not. """
    label_pred = model.predict(test_data)
    print("Accuracy: " + str(accuracy_score(test_label, label_pred)))
    print('\n')
    print(classification_report(test_label, label_pred))

def prepare_data(folder):
    """ Creates training and testing data for the SVM model. The training data consists of smaller extracts of the image
    containing the phone (positive images) or no phone (negative images). From these sub-images we extract HOG (Histogram oriented gradient)
    features to feed into the model."""

    # Get images from folder
    images_used_for_training = get_images(folder)

    # Create positive and negative images and immediately extract their hog features and get their labels
    hog_features, labels = get_training_data(folder, images_used_for_training)

    # Shuffle data
    data, label = shuffle(np.array(hog_features), np.array(labels), random_state=42)

    # Split data in training and testing data
    train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.2, random_state=42)

    return train_data, test_data, train_label, test_label

if __name__ == '__main__':
    ''' Main function '''
    image_folder = sys.argv[1]

    train_data, test_data, train_label, test_label = prepare_data(folder=image_folder)

    # Fit a SVM (Support Vector Machine) classifier to the data
    model = train_model(train_data, train_label)

    # Save the model
    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))

    # Test the model for accuracy
    # test_model(model, test_data, test_label)


