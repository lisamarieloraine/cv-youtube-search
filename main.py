# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 16:05:26 2018

@author: plagl
"""

# Import libraries
from data import Data
from network import CNN
import tensorflow as tf
import os
import json
import paths

# Setting up file names
ann_name_train = 'instances_train2017.json'
ann_name_val = 'instances_val2017.json'
file_name_train = 'annotations_train.json'
file_name_val = 'annotations_val.json'
info_name_train = 'info_train.json'
info_name_val = 'info_val.json'

# Setting up directories
# this is done in a separate module to avoid changing directories with every pull
ann_path, img_path_train, img_path_val = paths.init_paths()


if all(x in os.listdir(ann_path) for x in [info_name_train, info_name_val, file_name_train, file_name_val]):
    print('loading labels from json files...') 
    os.chdir( ann_path )
    with open(file_name_train) as t:
        classes_train = json.load(t)
    with open(file_name_val) as v:
        classes_val = json.load(v)
    with open(info_name_train) as t:
        info_train = json.load(t)
    with open(info_name_val) as v:
        info_val = json.load(v)
        
else:
    # Prepare training dataset
    data_train = Data(ann_path, ann_name_train, img_path_train)
    anns_train = data_train.get_annotations()
    info_train = data_train.get_info(anns_train, info_name_train)
    classes_train = data_train.convert_labels(anns_train, file_name_train)
    # Prepare validation dataset
    data_val = Data(ann_path, ann_name_val, img_path_val)
    anns_val = data_val.get_annotations()
    info_val = data_val.get_info(anns_val, info_name_val)
    classes_val = data_val.convert_labels(anns_val, file_name_val)


tf.reset_default_graph() 


# Set up the neural network
training_iters = 10
learning_rate = 0.001 
batch_size = 128
n_input = 28
n_classes = 3
network = CNN(training_iters, learning_rate, batch_size, n_input, n_classes, img_path_train, img_path_val)

# Train the network using the training data
train_loss, train_accuracy, val_loss, val_accuracy = network.train(info_train, classes_train, info_val, classes_val)
#network.plot_loss(train_loss, val_loss)
#network.plot_accuracy(train_accuracy, val_accuracy)



