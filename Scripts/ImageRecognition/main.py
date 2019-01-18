# Import libraries
import random
import imp
import data
imp.reload(data)
import network as net
imp.reload(net)
import tensorflow as tf
import os
import json
import paths
import numpy as np

# Entry point to our neural network
def run(write = False, predict = False, image = ""):
    
    tf.reset_default_graph() 
    klasses = [('banana', 0),('broccoli',1)]
    
    # Setting up directories
    # this is done in a separate module to avoid changing directories with every pull
    if predict == False:
        ann_path, img_path_train, img_path_val = paths.init_paths()
        #if we want to make a new selection of images
        if write:
            #banana and brocolli have a small intersection -> smaller problem
            data_object = data.Data(ann_path,klasses,img_path_train, img_path_val)
            data_object.write_data()
            
        #read in the selected images from file
        os.chdir(ann_path)
        with open('data_train.txt', 'r') as filehandle:  
            data_train = json.load(filehandle)   
        with open('data_val.txt', 'r') as filehandle:  
            data_val = json.load(filehandle)
            
        #validation set contains only few examples that are prominent enough
        #thats why we split the train set into a new train set and a validation set
        data_shuffle = data_train   
        random.shuffle(data_shuffle)
        data_val = data_shuffle[0:150]
        data_train = data_shuffle[150:]
        img_path_val = img_path_train 
    else:
        ann_path, img_path_train, img_path_val = "", "", ""
     
    
    # Setting up the neural network
    training_iters = 10
    learning_rate = 0.001  # 0.001 or 0.0001 are best
    batch_size = 1 # seems to have some influence on performance
    n_pixels = 64 # image resolution, i.e. 64x64 pixels
    n_classes = len(klasses)
    
    network = net.CNN(training_iters, learning_rate, batch_size, n_pixels, \
                  n_classes, img_path_train, img_path_val)
    
    # Either train the network or use a pretrained network for predicting
    if predict == False:
        # Create datasets for training and validation
        dataset_train,labels_train = data.create_dataset(data_train,batch_size,img_path_train,n_pixels)
        dataset_val,labels_val = data.create_dataset(data_val,batch_size,img_path_val,n_pixels)
        # Train the network using the training data
        train_loss, train_accuracy, val_loss, val_accuracy =\
        network.train(dataset_train,len(labels_train), dataset_val, len(labels_val))
        # Evaluate the performance of the network
        network.plot_loss(train_loss, val_loss)
        network.plot_accuracy(train_accuracy, val_accuracy)
        return
    else:   
        # Make a prediction for a given image
        result = network.predict(image)
        label = np.argmax(result)
        key = klasses[label][0]
        return key


#"D:\\cocoapi\\images\\test2017\\000000006610.jpg" broccoli
#"D:\\cocoapi\\images\\test2017\\000000013691.jpg" banana
#"D:\\cocoapi\\images\\test2017\\000000006054.jpg" banana
#"D:\\cocoapi\\images\\test2017\\000000007127.jpg" other
#"D:\\cocoapi\\images\\test2017\\000000014394.jpg" broccoli
search_term = run(write = False, predict = True, \
                  image = "D:\\cocoapi\\images\\test2017\\000000006054.jpg")
print(search_term)

