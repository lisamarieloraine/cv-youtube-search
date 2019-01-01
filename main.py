# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 16:05:26 2018

@author: plagl
"""

# Import libraries
from data import Data
from network import CNN


# Prepare training dataset
ann_path = "D:\\cocoapi\\annotations\\"
ann_name_train = "instances_train2017.json"
img_path_train = "D:\\cocoapi\\images\\train2017\\"
data_train = Data(ann_path, ann_name_train, img_path_train)
anns_train = data_train.get_annotations()
info_train = data_train.get_info(anns_train)
classes_train, counts_train = data_train.convert_labels(anns_train)
training_dataset = data_train.create_dataset(info_train, classes_train)


# Prepare testing dataset
ann_name_test = "instances_val2017.json"
img_path_test = "D:\\cocoapi\\images\\val2017\\"
data_test = Data(ann_path, ann_name_test, img_path_test)
anns_test = data_test.get_annotations()
info_test = data_test.get_info(anns_test)
classes_test, counts_test = data_test.convert_labels(anns_test)
testing_dataset = data_test.create_dataset(info_test, classes_test)


# Set up the neural network
training_iters = 100 
learning_rate = 0.001 
batch_size = 128
n_input = 28
n_classes = 3
network = CNN(training_iters, learning_rate, batch_size, n_input, n_classes)

# Train the network using the training data
#network.train(train_images, train_labels, test_images, test_labels)



