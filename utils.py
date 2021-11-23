import glob

def get_images(folder):
    """ Extract images from folder """
    image_paths = glob.glob(folder + '/*.jpg')
    return image_paths

def normalized_phone_coordinates(data_frame, image_file_name):
    """ Get true coordinate of phone """
    image_data_frame = data_frame[data_frame['image'] == image_file_name]
    x_norm = float(image_data_frame['x'])
    y_norm = float(image_data_frame['y'])
    return x_norm, y_norm

def from_window_coord_to_norm_coord(phone_position, image, half_window_side, half_window_size_normalized):
    """ Go from image indices to the normalized coordinates """

    x = phone_position[0]
    y = phone_position[1]

    height, width = image.shape

    # Normalize coordiantes
    if half_window_side is False:
        x_predict = x / width
        y_predict = y / height
    else:
        x_predict = (x/width) + half_window_size_normalized
        y_predict = (y/height) + half_window_size_normalized
    return x_predict, y_predict

def get_centroid(points):
    """ Compute centroid of several contour points """
    x = [p[0][0] for p in points]
    y = [p[0][1] for p in points]
    centroid = (sum(x) / len(points), sum(y) / len(points))
    return centroid