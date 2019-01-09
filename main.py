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

tf.reset_default_graph() 

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
    anns_train = data_train.create_dict(info_name_train, file_name_train)
    images_train, counts_train = data_train.convert_dict(anns_train, file_name_train)
    labels_train, file_names_train = data_train.subset_images(images_train, counts_train)
    dataset_train = data_train.create_dataset(file_names_train, labels_train, 128)
    #classes_train, counts_train = data_train.convert_labels(anns_train, file_name_train)
    #classes_train, info_train = data_train.subset_images(classes_train, counts_train, info_train, 'annotations_train_subset.json', 'info_train_subset.json')
    
    # Prepare validation dataset
    data_val = Data(ann_path, ann_name_val, img_path_val)
    anns_val = data_val.create_dict(info_name_val, file_name_val)
    images_val, counts_val = data_val.convert_dict(anns_val, file_name_val)
    labels_val, file_names_val = data_val.subset_images(images_val, counts_val)
    dataset_val = data_val.create_dataset(file_names_val, labels_val, 128)
    #classes_val, counts_val = data_val.convert_labels(anns_val, file_name_val)
    #classes_val, info_val = data_train.subset_images(classes_val, counts_val, info_val, 'annotations_val_subset.json', 'info_val_subset.json')



# Set up the neural network
training_iters = 10
learning_rate = 0.001 
batch_size = 128
n_input = 28
n_classes = 3
network = CNN(training_iters, learning_rate, batch_size, n_input, n_classes, img_path_train, img_path_val)

# Train the network using the training data
train_loss, train_accuracy, val_loss, val_accuracy = network.train(dataset_train, len(labels_train), dataset_val, len(labels_val))
network.plot_loss(train_loss, val_loss)
network.plot_accuracy(train_accuracy, val_accuracy)



