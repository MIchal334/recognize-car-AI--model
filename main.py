import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras import layers
from load_data import load_car_image_data
from tensorflow.keras.utils import to_categorical
import os
from tensorflow.python.keras import backend



def prepare_data():
    (images_data_train, images_data_test, labels_data_train,labels_data_test)= load_car_image_data()  
    print(images_data_test.shape)
    print(labels_data_train)
    images_data_train = images_data_train.astype('float32')/255
    images_data_test = images_data_test.astype('float32')/255
    labels_data_train = to_categorical(labels_data_train,2)
    labels_data_test = to_categorical(labels_data_test,2)
    return (images_data_train, labels_data_train), (images_data_test, labels_data_test)

def get_model(): 
  model = models.Sequential() 
  model.add(layers.Conv2D(128, (3,3), activation = keras.activations.relu, input_shape = train_data[0].shape)) 
  model.add(layers.MaxPool2D((2,2))) 
  model.add(layers.Conv2D(64, (3,3), activation = keras.activations.selu)) 
  model.add(layers.MaxPool2D((2,2))) 
  model.add(layers.Conv2D(32, (3,3), activation = keras.activations.elu)) 
  model.add(layers.Flatten()) 
  model.add(layers.Dense(512, activation = keras.activations.selu)) 
  model.add(layers.Dropout(0.2))
  model.add(layers.Dense(256,activation=keras.activations.elu))
  model.add(layers.Dense(64,activation=keras.activations.elu))
  model.add(layers.Dense(2, activation = keras.activations.sigmoid))
  # model.summary()
  model.compile(optimizer= keras.optimizers.Adam(), loss = keras.losses.BinaryCrossentropy(from_logits=True), metrics = keras.metrics.AUC())
  return model



def train_network(train_data,train_labels,test_data,test_labels):
    network = get_model()
    network.fit(train_data,train_labels,batch_size = 16, epochs=15)




if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
    backend.set_session(sess)

    (train_data, train_labels), (test_data, test_labels) = prepare_data()
    train_network(train_data,train_labels,test_data,test_labels)
