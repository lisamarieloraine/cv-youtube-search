# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 14:47:15 2018

@author: plagl
"""

# Import modules
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" #for training on gpu


class CNN:
    def __init__(self, epochs, alpha, batch_size, n_inputs, n_classes, img_dir_train, img_dir_val):
        self.epochs = epochs
        self.learning_rate = alpha
        self.batch_size = batch_size
        # input data shape (img shape: 28*28)
        self.n_input = n_inputs
        # total classes (0-3 digits)
        self.n_classes = n_classes
        self.img_dir_train = img_dir_train
        self.img_dir_val = img_dir_val

        #placeholders for images and labels
        #self.placeholder_X = tf.placeholder(tf.float32, [None, n_inputs, n_inputs, 1])
        #self.placeholder_y = tf.placeholder(tf.int32, [None])
        
        #set up dictionaries for the structure of the weight and bias terms
        self.weights = {
        'wc1': tf.get_variable('w0', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()), 
        'wc2': tf.get_variable('w1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()), 
        'wc3': tf.get_variable('w2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()), 
        'wd1': tf.get_variable('w3', shape=(4*4*128,128), initializer=tf.contrib.layers.xavier_initializer()), 
        'out': tf.get_variable('w6', shape=(128,n_classes), initializer=tf.contrib.layers.xavier_initializer()), 
        }
        self.biases = {
            'bc1': tf.get_variable('b0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
            'bc2': tf.get_variable('b1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
            'bc3': tf.get_variable('b2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
            'bd1': tf.get_variable('b3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
            'out': tf.get_variable('b4', shape=(n_classes), initializer=tf.contrib.layers.xavier_initializer()),
        }
     
        
    @staticmethod
    def plot_loss(train_loss, test_loss):
        plt.plot(range(len(train_loss)), train_loss, 'b', label='Training loss')
        plt.plot(range(len(test_loss)), test_loss, 'r', label='Test loss')
        plt.title('Training and Test loss')
        plt.xlabel('Epochs ',fontsize=16)
        plt.ylabel('Loss',fontsize=16)
        plt.legend()
        plt.figure()
        plt.show()
        return
    
    
    @staticmethod
    def plot_accuracy(train_accuracy, test_accuracy):
        plt.plot(range(len(train_accuracy)), train_accuracy, 'b', label='Training Accuracy')
        plt.plot(range(len(test_accuracy)), test_accuracy, 'r', label='Test Accuracy')
        plt.title('Training and Test Accuracy')
        plt.xlabel('Epochs ',fontsize=16)
        plt.ylabel('Loss',fontsize=16)
        plt.legend()
        plt.figure()
        plt.show()
        return
    
        
    @staticmethod    
    def conv2d(x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x) 
    
    
    @staticmethod  
    def maxpool2d(x, k=2):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')
    
    
    def conv_net(self, x):  
        # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
        conv1 = self.conv2d(x, self.weights['wc1'], self.biases['bc1'])
        # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
        conv1 = self.maxpool2d(conv1, k=2)
        
        # Convolution Layer
        # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
        conv2 = self.conv2d(conv1, self.weights['wc2'], self.biases['bc2'])
        # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix.
        conv2 = self.maxpool2d(conv2, k=2)
    
        conv3 = self.conv2d(conv2,self.weights['wc3'], self.biases['bc3'])
        # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
        conv3 = self.maxpool2d(conv3, k=2)
    
        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv3, [-1, self.weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, self.weights['wd1']), self.biases['bd1'])
        fc1 = tf.nn.relu(fc1)

        # Output, class prediction
        # finally we multiply the fully connected layer with the weights and add a bias term. 
        out = tf.add(tf.matmul(fc1, self.weights['out']), self.biases['out'])
        return out
    
    
    def operations(self, x, y):
        pred = self.conv_net(x)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
    
        #Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image. and both will be a column vector.
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        #calculate accuracy across all the given images and average them out. 
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return optimizer, accuracy, cost
    
    
    #def train(self, info_train, anns_train, info_val, anns_val):
    def train(self, info_train, anns_train, info_val, anns_val):
        # implements a reinitializable iterator
        # see https://medium.com/ymedialabs-innovation/how-to-use-dataset-and-iterators-in-tensorflow-with-code-samples-3bb98b6b74ab
        print("training the network...")
        
        # Reads an image from a file, decodes it into a dense tensor, and resizes it
        # to a fixed shape.
        def _parse_function(filename, label):
          image_string = tf.read_file(filename)
          image_decoded = tf.image.decode_jpeg(image_string, channels=1)
          image_resized = tf.image.resize_images(image_decoded, [28, 28])
          return image_resized, label
       #ValueError: Shape must be rank 0 but is rank 1 for 'ReadFile' (op: 'ReadFile') with input shapes: [?].
        print('creating datasets...')
        # A vector of filenames.
        filenames_train = tf.constant([f.get('file_name') for f in info_train])
        filenames_val = tf.constant([f.get('file_name') for f in info_val])
        # `labels[i]` is the label for the image in `filenames[i].
        labels_train = tf.constant([f.get('category_id') for f in anns_train])
        labels_val = tf.constant([f.get('category_id') for f in anns_val])
        
        #might improve efficiency for large datasets
        #placeholder_X = tf.placeholder(filenames_train.dtype, filenames_train.shape)
        #placeholder_y = tf.placeholder(labels_train.dtype, labels_train.shape)
        
        # Create separate Datasets for training and validation
        os.chdir( self.img_dir_train )
        train_dataset = tf.data.Dataset.from_tensor_slices((filenames_train, labels_train))
        train_dataset = train_dataset.map(_parse_function).batch(self.batch_size)
        os.chdir( self.img_dir_val )
        val_dataset = tf.data.Dataset.from_tensor_slices((filenames_val, labels_val))
        val_dataset = val_dataset.map(_parse_function).batch(self.batch_size)
        print('datasets created!')
        
        # Iterator has to have same output types across all Datasets to be used
        iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        data_X, data_y = iterator.get_next()
        data_y = tf.cast(data_y, tf.int32)
        #data_y = tf.one_hot(data_y, depth = self.n_classes)
        optimizer, accuracy, cost = self.operations(data_X, data_y)
        
        # Initialize with required Datasets
        train_iterator = iterator.make_initializer(train_dataset)
        val_iterator = iterator.make_initializer(val_dataset)
        
        # Initializing the variables and operations
        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            sess.run(init) 
            train_losses = []
            val_losses = []
            train_accuracies = []
            val_accuracies = []
            summary_writer = tf.summary.FileWriter('./Output', sess.graph)
            for i in range(self.epochs):
                train_loss, train_accuracy = 0, 0
                val_loss, val_accuracy = 0, 0
                # Start train iterator
                sess.run(train_iterator)
                try:
                    while True:
                        opt, acc, loss = sess.run([optimizer, accuracy, cost])
                        train_loss += loss
                        train_accuracy += acc
                except tf.errors.OutOfRangeError:
                    pass
                
                # Start validation iterator
                sess.run(val_iterator)
                try:
                    while True:
                        acc, loss = sess.run([accuracy, cost])
                        val_loss += loss
                        val_accuracy += acc
                except tf.errors.OutOfRangeError:
                    pass
        
                print('\nEpoch: {}'.format(i + 1))
                print('Train accuracy: {:.4f}, loss: {:.4f}'.format(train_accuracy / len(labels_train),
                                                                     train_loss / len(labels_train)))
                print('Val accuracy: {:.4f}, loss: {:.4f}\n'.format(val_accuracy / len(labels_val), 
                                                                    val_loss / len(labels_val)))


                # Append data of current epoch
                train_losses.append(train_loss / len(labels_train))
                val_losses.append(val_loss / len(labels_val))
                train_accuracies.append(train_accuracy / len(labels_train))
                val_accuracies.append(val_accuracy / len(labels_val))
        
            summary_writer.close()
        return train_losses, train_accuracies, val_losses, val_accuracies
    





