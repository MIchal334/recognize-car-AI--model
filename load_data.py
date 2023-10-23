import os
import numpy as np
import cv2


dictionar_with_car_image_path = '/home/michal/Desktop/photos/data/vehicles'
dictionar_with_normal_image_path = '/home/michal/Desktop/photos/data/non-vehicles'
image_file_extend = ('.png')
x_size = 512
y_size = 512

def load_car_image_data():
    images_data = []
    labels_data = []
    (car_image, car_labels) = __load_image_from_file_by_path(dictionar_with_car_image_path,1)
    (no_car_image, no_car_labels) = __load_image_from_file_by_path(dictionar_with_normal_image_path,1)
    
    images_data.extend(car_image)
    images_data.extend(no_car_image)
    labels_data.extend(car_labels)
    labels_data.extend(no_car_labels)
    
    return (np.array(images_data), np.array(labels_data))




def __load_image_from_file_by_path(path: str, label: int):
    images_pixels = []
    labels = []
    for img in os.listdir(path):
        if img.lower().endswith(image_file_extend):
            img_path = os.path.join(path, img)
            img_pix = cv2.imread(img_path, 1)
            img_pix = cv2.resize(img_pix, (x_size, y_size))  
            images_pixels.append(img_pix)
            labels.append(label)
    return (images_pixels,labels)