import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

dictionar_with_car_image_path = '/home/michal/Desktop/photos/data/vehicles'
dictionar_with_normal_image_path = '/home/michal/Desktop/photos/data/non-vehicles'
image_file_extend = ('.png')
x_size = 64
y_size = 64
frequency = 5

def load_car_image_data():
    images_data_train = []
    labels_data_train = []
    images_data_test = []
    labels_data_test = []
    (car_image_train, car_image_test, car_labels_train, car_labels_test) = __load_image_from_file_by_path(dictionar_with_car_image_path,1)
    (no_car_image_train, no_car_image_test, no_car_labels_train, no_car_labels_test) = __load_image_from_file_by_path(dictionar_with_normal_image_path,0)
    
    images_data_train.extend(car_image_train)
    images_data_train.extend(no_car_image_train)
    labels_data_train.extend(car_labels_train)
    labels_data_train.extend(no_car_labels_train)

    images_data_test.extend(car_image_test)
    images_data_test.extend(no_car_image_test)
    labels_data_test.extend(car_labels_test)
    labels_data_test.extend(no_car_labels_test)
    
    return (np.array(images_data_train), np.array(images_data_test), np.array(labels_data_train),np.array(labels_data_test))




def __load_image_from_file_by_path(path: str, label: int):
    images_train = []
    labels_train = []
    images_test = []
    labels_test = []
    itterator  = 0 
    for img in os.listdir(path):
        if img.lower().endswith(image_file_extend):
            if itterator % frequency == 0:  
                images_test.append(__load_cv2_image(path,img))
                labels_test.append(label)
                itterator = itterator + 1
                continue
            images_train.append(__load_cv2_image(path,img))
            labels_train.append(label)
            itterator = itterator + 1
    return (images_train,images_test,labels_train,labels_test)


def __load_cv2_image(path,img):
    img_path = os.path.join(path, img)
    img_pix = cv2.imread(img_path, 1)
    return cv2.resize(img_pix, (x_size, y_size))  