# Import libraries
import random
import imp
import data
imp.reload(data)
import network
imp.reload(network)
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
    #banana and brocolli have a small intersection -> toy problem
    data_object = data.Data(ann_path,klasses,img_path_train, img_path_val)
    data_object.write_data()
    
os.chdir(ann_path)
#read in the selected images from file
with open('data_train.txt', 'r') as filehandle:  
    data_train = json.load(filehandle)
    
with open('data_val.txt', 'r') as filehandle:  
    data_val = json.load(filehandle)


#validation set contains few examples that are prominent enough
#thats why i split the train set into a new train set and a validation set
data_shuffle = data_train   
random.shuffle(data_shuffle)
data_val = data_shuffle[0:200]
data_train = data_shuffle[200:]
img_path_val = img_path_train  


    
# Set up the neural network
training_iters = 20
learning_rate = 0.001 
#0.001 or 0.0001 are best but overfitting is an issue
batch_size = 1 #seems to have a big influence, not sure whats wrong if i increase
n_input = 32
n_classes = 2

dataset_train,labels_train = data.create_dataset(data_train,batch_size,img_path_train)
dataset_val,labels_val = data.create_dataset(data_val,batch_size,img_path_val)

network = network.CNN(training_iters, learning_rate, batch_size, n_input, \
              n_classes, img_path_train, img_path_val)

#Train the network using the training data
train_loss, train_accuracy, val_loss, val_accuracy =\
network.train(dataset_train,len(labels_train), dataset_val, len(labels_val))

network.plot_loss(train_loss, val_loss)
network.plot_accuracy(train_accuracy, val_accuracy)



