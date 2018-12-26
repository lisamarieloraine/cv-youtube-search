# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 16:05:26 2018

@author: plagl
"""

# Import libraries
from data import Data
from network import CNN
import json


# Prepare datasets
ann_path = "D:\\cocoapi\\annotations\\"
img_path = "D:\\cocoapi\\images\\"
data = Data(ann_path, img_path)
anns = data.get_annotations()
info = data.get_info(anns)
# classes, counts = data.convert_labels(anns)
classes = json.loads(open('annotations.json').read())
dataset = data.create_dataset(info, classes)
# img = data.load_images(info)


# Set up the neural network
training_iters = 200 
learning_rate = 0.001 
batch_size = 128
n_input = 28
n_classes = 3
network = CNN(training_iters, learning_rate, batch_size, n_input, n_classes)



