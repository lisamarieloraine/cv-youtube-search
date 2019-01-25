# Import libraries
import random
import imp
import Scripts.ImageRecognition.data as data
imp.reload(data)
import Scripts.ImageRecognition.network as net
imp.reload(net)
import tensorflow as tf
import os
import json
import Scripts.ImageRecognition.paths as paths
import numpy as np



# Entry point to our neural network
def run(write=False, predict=False, image=""):
    tf.reset_default_graph()
    klasses = [('banana', 0), ('broccoli', 1)]

    if predict == False:
        # Setting up directories:
        # This is done in a separate module to avoid changing directories with every pull.
        # If the paths.py file is not added to the repo, everyone has to create this file 
        # manually and define the paths where the training and validation images as well
        # as the annotations are stored on his PC.
        ann_path, img_path_train, img_path_val = paths.init_paths()
        
        # if write = True, we want to make a new selection of images
        if write:
            # banana and brocolli have a small intersection -> smaller problem
            data_object = data.Data(ann_path, klasses, img_path_train, img_path_val)
            data_object.write_data()

        # read in the selected images from file
        os.chdir(ann_path)
        with open('data_train.txt', 'r') as filehandle:
            data_train = json.load(filehandle)
        with open('data_val.txt', 'r') as filehandle:
            data_val = json.load(filehandle)

        # validation set contains only few examples that are prominent enough
        # thats why we split the train set into a new train set and a validation set
        data_shuffle = data_train   
        validation = int(len(data_shuffle)/8) 
        random.shuffle(data_shuffle)
        data_val = data_shuffle[:validation]
        data_train = data_shuffle[validation:]
        img_path_val = img_path_train
    else:
        # paths not needed for making predictions, only for training
        ann_path, img_path_train, img_path_val = "", "", ""

    # Setting up the neural network
    training_iters = 10
    learning_rate = 0.001  # 0.001 or 0.0001 are best
    batch_size = 1  # seems to have some influence on performance
    n_pixels = 64  # image resolution, i.e. 64x64 pixels
    n_classes = len(klasses)

    network = net.CNN(training_iters, learning_rate, batch_size, n_pixels, \
                      n_classes, img_path_train, img_path_val)
    script_path = os.getcwd()
    # Either train the network or use a pretrained network for predicting
    if predict == False:
        # Create datasets for training and validation
        dataset_train, labels_train = data.create_dataset(data_train, batch_size, img_path_train, n_pixels)
        dataset_val, labels_val = data.create_dataset(data_val, batch_size, img_path_val, n_pixels)
        # Train the network using the training data
        train_loss, train_accuracy, val_loss, val_accuracy = \
            network.train(dataset_train, len(labels_train), dataset_val, len(labels_val))
        # Evaluate the performance of the network
        network.plot_loss(train_loss, val_loss)
        network.plot_accuracy(train_accuracy, val_accuracy)
        return
    else:
        # Make a prediction for a given image, also supporting a third 'None' type
        result = network.predict(image)
        if result[0][0] > 0.5 and result[0][1] < -0.5 and result[0][1] > -2.4:
            key = "banana"
        elif result[0][0] < -0.5 and result[0][1] > -0.5  :
            key = "broccoli"
        else:
            key = "object not supported"
        
        #label = np.argmax(result)
        #key = klasses[label][0]
        os.chdir( script_path )
        return key


