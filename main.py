# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 16:05:26 2018

@author: plagl
"""

# Import libraries
from data import Data
from network import CNN
import tensorflow as tf


# Prepare training dataset
ann_path = "D:\\cocoapi\\annotations\\"
ann_name_train = "instances_train2017.json"
img_path_train = "D:\\cocoapi\\images\\train2017\\"
data_train = Data(ann_path, ann_name_train, img_path_train)
anns_train = data_train.get_annotations()
info_train = data_train.get_info(anns_train)
classes_train, counts_train = data_train.convert_labels(anns_train)

# Prepare validation dataset
ann_name_val = "instances_val2017.json"
img_path_val = "D:\\cocoapi\\images\\val2017\\"
data_val = Data(ann_path, ann_name_val, img_path_val)
anns_val = data_val.get_annotations()
info_val = data_val.get_info(anns_val)
classes_val, counts_val = data_val.convert_labels(anns_val)

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



