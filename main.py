# Import libraries
from data import Data
from data import create_dataset
from network import CNN
import tensorflow as tf
import os
import json
import paths

tf.reset_default_graph() 

# Setting up directories
# this is done in a separate module to avoid changing directories with every pull
ann_path, img_path_train, img_path_val = paths.init_paths()

#if we want to make a new selection of images
write = False
if write:
    klasses = [('banana', 0),('broccoli',1)] 
    #banana and brocolli have a small union -> toy problem
    data_object = Data(ann_path,klasses,img_path_train, img_path_val)
    data_object.write_data()
    
os.chdir(ann_path)

with open('data_train.txt', 'r') as filehandle:  
    data_train = json.load(filehandle)
    
with open('data_val.txt', 'r') as filehandle:  
    data_val = json.load(filehandle)
    

# Set up the neural network
training_iters = 20
learning_rate = 0.001 
batch_size = 8
n_input = 28
n_classes = 2

dataset_train,labels_train = create_dataset(data_train,batch_size,img_path_train)
dataset_val,labels_val = create_dataset(data_val,batch_size,img_path_val)

network = CNN(training_iters, learning_rate, batch_size, n_input, \
              n_classes, img_path_train, img_path_val)

#Train the network using the training data
train_loss, train_accuracy, val_loss, val_accuracy =\
network.train(dataset_train,len(labels_train), dataset_val, len(labels_val))

network.plot_loss(train_loss, val_loss)
network.plot_accuracy(train_accuracy, val_accuracy)



