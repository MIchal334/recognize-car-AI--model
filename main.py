import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras import layers
from load_data import load_car_image_data
from tensorflow.keras.utils import to_categorical
import os



def prepare_data():
    (train_data, train_labels) = load_car_image_data()  
    print(train_data.shape)
    # print(train_labels)
    # train_data = train_data.astype('float32')/255
    # test_data = test_data.astype('float32')/255
    # train_labels = to_categorical(train_labels,10)
    # test_labels = to_categorical(test_labels,10)
    # return (train_data, train_labels), (test_data, test_labels)

# def get_model(): 
#   model = models.Sequential() 
#   model.add(layers.Conv2D(128, (3,3), activation = keras.activations.relu, input_shape = train_data[0].shape)) 
#   model.add(layers.MaxPool2D((2,2))) 
#   model.add(layers.Conv2D(64, (3,3), activation = keras.activations.selu)) 
#   model.add(layers.MaxPool2D((2,2))) 
#   model.add(layers.Conv2D(32, (3,3), activation = keras.activations.elu)) 
#   model.add(layers.Flatten()) 
#   model.add(layers.Dense(512, activation = keras.activations.selu)) 
#   model.add(layers.Dropout(0.2))
#   model.add(layers.Dense(256,activation=keras.activations.elu))
#   model.add(layers.Dense(64,activation=keras.activations.elu))
#   model.add(layers.Dense(10, activation = keras.activations.sigmoid))
#   # model.summary()
#   model.compile(optimizer= keras.optimizers.Adam(), loss = keras.losses.categorical_crossentropy, metrics = keras.metrics.AUC())
#   return model



# def train_network(train_data,train_labels,test_data,test_labels):
#     network = get_model()
#     network.fit(train_data,train_labels, batch_size=64, epochs=15, validation_data=(test_data,test_labels))




if __name__ == "__main__":
    prepare_data()
    # (train_data, train_labels), (test_data, test_labels) = prepare_data()
    # train_network(train_data,train_labels,test_data,test_labels)
