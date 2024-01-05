import os
import keras
from keras.models import load_model
import cv2
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np

# path_to_test_dictionary = '/home/michal/Desktop/photos/car_boxes/test_set'
path_to_test_dictionary = '/home/michal/Desktop/photos/auto_test'
# path_to_test_dictionary = '/home/michal/Desktop/photos/non_auto_test'

is_car_test = 1


def show_test_set(network):
    test_set = __load_images()
 
    bad_amount = 0
    for img in test_set:
        image_prepared = __preapre_image_for_proccesing(img)
        result = network.predict(image_prepared)
        class_max = np.argmax(result)
        if class_max != is_car_test:
            plt.imshow(img)
            plt.show()
            bad_amount = bad_amount + 1
        print(class_max)

    print(f'BAD:{bad_amount}')


def __preapre_image_for_proccesing(img):
    x_size = 64
    y_size = 64
    image = cv2.resize(img, (x_size, y_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image[np.newaxis, ...]
    return image.astype('float32')/255


def __load_images():
    images = []
    for filename in os.listdir(path_to_test_dictionary):
        img_path = os.path.join(path_to_test_dictionary, filename)
        images.append(cv2.imread(img_path, 1))
    return images


if __name__ == "__main__":
    network = keras.models.load_model('car_recognize_model.keras')
    show_test_set(network)
