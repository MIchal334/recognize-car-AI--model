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
from check_history import show_all_history

checkpoint_filepath = './tmp/checkpoint_recognize'


def prepare_data():
    (images_data_train, images_data_test, labels_data_train,labels_data_test)= load_car_image_data()  
    images_data_train = images_data_train.astype('float32')/255
    images_data_test = images_data_test.astype('float32')/255
    labels_data_train = to_categorical(labels_data_train,2)
    labels_data_test = to_categorical(labels_data_test,2)
    train_data_expanded = np.expand_dims(images_data_train, axis=-1)
    test_data_expanded = np.expand_dims(images_data_test, axis=-1)
    return (train_data_expanded, labels_data_train), (test_data_expanded, labels_data_test)

def get_model(): 
  model = models.Sequential() 
  model.add(layers.Conv2D(64, (4,4), activation = keras.activations.relu, input_shape = (64,64,1), padding='same'))  
  model.add(layers.Conv2D(64, (4,4), activation = keras.activations.relu, input_shape = (64,64,1), padding='same'))  
  model.add(layers.MaxPool2D((3,3))) 
  model.add(layers.Conv2D(32, (3,3), activation = keras.activations.elu, padding='same')) 
  model.add(layers.Conv2D(32, (3,3), activation = keras.activations.elu, padding='same')) 
  model.add(layers.MaxPool2D((2,2))) 
  model.add(layers.Dropout(0.5))
  model.add(layers.Conv2D(16, (2,2), activation = keras.activations.relu, padding='same')) 
  model.add(layers.Conv2D(16, (2,2), activation = keras.activations.relu, padding='same')) 
  model.add(layers.MaxPool2D((2,2))) 
  
  model.add(layers.Flatten())
  model.add(layers.Dense(32,activation=keras.activations.elu))
  model.add(layers.Dense(16,activation=keras.activations.relu))
  model.add(layers.Dense(8,activation=keras.activations.relu))
  model.add(layers.Dense(4,activation=keras.activations.relu))
  model.add(layers.Dense(2, activation = keras.activations.sigmoid))
  model.summary()
  model.compile(optimizer=keras.optimizers.Adam(), 
              loss=keras.losses.BinaryCrossentropy(), 
              metrics=['accuracy'])
  return model



def train_network(train_data,train_labels,test_data,test_labels):
    network = get_model()
    callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                              save_weights_only=True,
                                              monitor='val_accuracy',
                                              mode='max',
                                              verbose=1,
                                              save_best_only=True)
    return network.fit(train_data,train_labels,batch_size = 20,
    validation_data=(test_data,test_labels), epochs=50, callbacks=[callback])




if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
    backend.set_session(sess)

    (train_data, train_labels), (test_data, test_labels) = prepare_data()
    history = train_network(train_data,train_labels,test_data,test_labels)
    show_all_history(history)


    model = get_model()
    model.load_weights(checkpoint_filepath)
    model.evaluate(test_data,test_labels)

    model.save('car_recognize_model.keras')
