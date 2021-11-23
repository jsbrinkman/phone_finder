import pandas as pd
import pickle
from find_phone import detect_phone
from train_phone_finder import normalized_phone_coordinates, get_images
import os
import numpy as np

def accuracy(folder, folder_with_images):
    """ Computes accuracy of the trained model on the training set."""

    data_frame = pd.read_table(folder + '/labels.txt', sep=' ', names=['image', 'x', 'y'])
    total = 0
    accuracy = 0

    filename = 'finalized_model.sav'
    model = pickle.load(open(filename, 'rb'))

    for image_path in folder_with_images:
        total = total + 1
        x_predict, y_predict = detect_phone(image_path, model)
        image_file_name = os.path.basename(os.path.normpath(image_path))
        x_true, y_true = normalized_phone_coordinates(data_frame, image_file_name)
        res = np.sqrt((x_predict - x_true)**2 + (y_predict-y_true)**2)

        # Check if the predicted coordinate is within 0.05 radius of the true coordinate
        if res <= 0.05:
            accuracy = accuracy + 1
            print("accurate")
        else:
            print("not accurate")
    accuracy = accuracy / total
    print(accuracy)

    return accuracy

if __name__ == '__main__':
    image_folder = 'find_phone'
    folder_with_images = get_images(image_folder)
    accuracy(image_folder, folder_with_images)