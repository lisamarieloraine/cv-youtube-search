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
    def __init__(self, epochs, alpha, batch_size, n_inputs, n_classes):
        self.training_iters = epochs
        self.learning_rate = alpha
        self.batch_size = batch_size
        # input data shape (img shape: 28*28)
        self.n_input = n_inputs
        # total classes (0-3 digits)
        self.n_classes = n_classes

        #both placeholders are of type float
        self.image = tf.placeholder("float", [None, n_inputs,n_inputs,1])
        self.labels = tf.placeholder("float", [None, n_classes])
        
        #set up dictionaries for the structure of the weight and bias terms
        self.weights = {
        'wc1': tf.get_variable('W0', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()), 
        'wc2': tf.get_variable('W1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()), 
        'wc3': tf.get_variable('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()), 
        'wd1': tf.get_variable('W3', shape=(4*4*128,128), initializer=tf.contrib.layers.xavier_initializer()), 
        'out': tf.get_variable('W6', shape=(128,n_classes), initializer=tf.contrib.layers.xavier_initializer()), 
        }
        self.biases = {
            'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
            'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
            'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
            'bd1': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
            'out': tf.get_variable('B4', shape=(10), initializer=tf.contrib.layers.xavier_initializer()),
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
    
    
    def conv_net(self):  
        # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
        conv1 = self.conv2d(self.image, self.weights['wc1'], self.biases['bc1'])
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
    
    
    def operations(self):
        pred = self.conv_net(self.image, self.weights, self.biases)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.labels))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
    
        #Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image. and both will be a column vector.
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        #calculate accuracy across all the given images and average them out. 
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return cost, optimizer, accuracy
    
    
    def train(self, train_X, train_y, test_X, test_y):
        print("training the network...")
        # Initializing the variables and operations
        init = tf.global_variables_initializer()
        cost, optimizer, accuracy = self.operations()
        
        with tf.Session() as sess:
            sess.run(init) 
            train_loss = []
            test_loss = []
            train_accuracy = []
            test_accuracy = []
            summary_writer = tf.summary.FileWriter('./Output', sess.graph)
            for i in range(self.training_iters):
                for batch in range(len(train_X)//self.batch_size):
                    batch_x = train_X[batch*self.batch_size:min((batch+1)*self.batch_size,len(train_X))]
                    batch_y = train_y[batch*self.batch_size:min((batch+1)*self.batch_size,len(train_y))]    
                    # Run optimization op (backprop).
                    # Calculate batch loss and accuracy
                    opt = sess.run(optimizer, feed_dict={self.image: batch_x, self.labels: batch_y})
                    loss, acc = sess.run([cost, accuracy], feed_dict={self.image: batch_x, self.labels: batch_y})
                print("Iter " + str(i) + ", Loss= " + \
                              "{:.6f}".format(loss) + ", Training Accuracy= " + \
                              "{:.5f}".format(acc))
                print("Optimization Finished!")
                
        # Calculate accuracy for all test images
        test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: test_X, y: test_y})
        train_loss.append(loss)
        test_loss.append(valid_loss)
        train_accuracy.append(acc)
        test_accuracy.append(test_acc)
        print("Testing Accuracy:","{:.5f}".format(test_acc))
        
        summary_writer.close()
        return train_loss, train_accuracy, test_loss, test_accuracy
    

