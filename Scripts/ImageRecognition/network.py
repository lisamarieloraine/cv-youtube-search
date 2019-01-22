# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 14:47:15 2018

@author: plagl
"""

# Import modules
import matplotlib.pyplot as plt
import tensorflow as tf
import os


class CNN:
    def __init__(self, epochs, alpha, batch_size, n_inputs, n_classes, img_dir_train, img_dir_val):
        self.epochs = epochs
        self.learning_rate = alpha
        self.batch_size = batch_size
        self.n_input = n_inputs # input data shape (image shape: 64*64 pixels)
        self.n_classes = n_classes # total number of classes
        self.img_dir_train = img_dir_train
        self.img_dir_val = img_dir_val
        
        #set up dictionaries for the structure of the weight and bias terms
        self.weights = {
        'wc1_1': tf.get_variable('w1_1', shape=(3,3,3,16), initializer=tf.contrib.layers.xavier_initializer()), 
        'wc1_2' :tf.get_variable('w1_2', shape=(3,3,16,16), initializer=tf.contrib.layers.xavier_initializer()), 
        
        'wc2_1': tf.get_variable('w2_1', shape=(3,3,16,32), initializer=tf.contrib.layers.xavier_initializer()), 
        'wc2_2': tf.get_variable('w2_2', shape=(3,3,32,32), initializer=tf.contrib.layers.xavier_initializer()), 
        
        'wd1_1': tf.get_variable('wd1_1', shape=(13*13*32,32), initializer=tf.contrib.layers.xavier_initializer()), 
        
        'out': tf.get_variable('w6', shape=(32,n_classes), initializer=tf.contrib.layers.xavier_initializer()), 
        }
        
        self.biases = {
            'bc1_1': tf.get_variable('b1_1', shape=(16), initializer=tf.contrib.layers.xavier_initializer()),
            'bc1_2': tf.get_variable('b1_2', shape=(16), initializer=tf.contrib.layers.xavier_initializer()),
            
            'bc2_1': tf.get_variable('b2_1', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
            'bc2_2': tf.get_variable('b2_2', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
            
            'bd1_1': tf.get_variable('bd1_1', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
            
            'out': tf.get_variable('b4', shape=(n_classes), initializer=tf.contrib.layers.xavier_initializer()),
        }
    
    
    def conv_net(self, x,train = False):  

        conv1 = self.conv2d(x, self.weights['wc1_1'], self.biases['bc1_1'])   #62
        conv1 = self.conv2d(conv1,self.weights['wc1_2'], self.biases['bc1_2'])   #60
        conv1 = self.maxpool2d(conv1,k=2) #30
        
        conv2 = self.conv2d(conv1,self.weights['wc2_1'], self.biases['bc2_1']) ##28
        conv2 = self.conv2d(conv2,self.weights['wc2_2'], self.biases['bc2_2']) ##26
        conv2 = self.maxpool2d(conv2,k=2) #13

        
        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, self.weights['wd1_1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, self.weights['wd1_1']), self.biases['bd1_1'])
        fc1 = tf.nn.relu(fc1)
        # add dropout layer to reduce overfitting
        fc1 = tf.layers.dropout(fc1,rate = 0.7,training = train)
        
        # Output, class prediction
        # finally we multiply the fully connected layer with the weights and add a bias term. 
        out = tf.add(tf.matmul(fc1, self.weights['out']), self.biases['out'])
        return out

    @staticmethod
    def plot_loss(train_loss, test_loss):
        plt.plot(range(len(train_loss)), train_loss, 'b', label='Training loss')
        plt.plot(range(len(test_loss)), test_loss, 'r', label='Validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs ',fontsize=16)
        plt.ylabel('Loss',fontsize=16)
        plt.legend()
        plt.figure()
        plt.show()
        return
    
    
    @staticmethod
    def plot_accuracy(train_accuracy, test_accuracy):
        plt.plot(range(len(train_accuracy)), train_accuracy, 'b', label='Training Accuracy')
        plt.plot(range(len(test_accuracy)), test_accuracy, 'r', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs ',fontsize=16)
        plt.ylabel('Accuracy',fontsize=16)
        plt.legend()
        plt.figure()
        plt.show()
        return
    
        
    @staticmethod    
    def conv2d(x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x) 
    
    
    @staticmethod  
    def maxpool2d(x, k=2):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='VALID')

    
    def operations(self, x, y):
        pred = self.conv_net(x,False)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
        pred_dropout = self.conv_net(x,True)
        cost_optimizer = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_dropout, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,epsilon = 0.1).minimize(cost_optimizer)
    
        #Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image. and both will be a column vector.
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        #calculate accuracy across all the given images and average them out. 
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return optimizer, accuracy, cost
    
    
    def train(self, train_dataset, size_train, val_dataset, size_val):
        # implements a reinitializable iterator
        # see https://medium.com/ymedialabs-innovation/how-to-use-dataset-and-iterators-in-tensorflow-with-code-samples-3bb98b6b74ab
        print("training the network...")
        
        # Iterator has to have same output types across all Datasets to be used
        iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        data_X, data_y = iterator.get_next()
        data_y = tf.cast(data_y, tf.int32)
        data_y = tf.one_hot(data_y, depth = self.n_classes)
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
            
            epoch = 0
            
            val_loss_check = 1
            while val_loss_check > 0.43 and epoch < self.epochs:
                # Initialize performance measures
                train_loss, train_accuracy = 0, 0
                val_loss, val_accuracy = 0, 0
                
                # Start train iterator
                os.chdir( self.img_dir_train )
                sess.run(train_iterator)
                try:
                    while True:
                        opt, acc, loss = sess.run([optimizer, accuracy, cost])
                        train_loss += loss
                        train_accuracy += acc
                except tf.errors.OutOfRangeError:
                    pass
                
                # Start validation iterator
                os.chdir( self.img_dir_val )
                sess.run(val_iterator)
                try:
                    while True:
                        acc, loss = sess.run([accuracy, cost])
                        val_loss += loss
                        val_accuracy += acc
                except tf.errors.OutOfRangeError:
                    pass
                
                # Udpate epoch info
                #val_acc = (val_accuracy / size_val)
                epoch += 1
                val_loss_check = val_loss / size_val
                # Append data of current epoch
                print('\nEpoch: {}'.format(epoch),\
                      'Train accuracy: {:.4f}, loss: {:.4f}'.format(train_accuracy / size_train,train_loss / size_train),\
                      'Val accuracy: {:.4f}, loss: {:.4f}\n'.format(val_accuracy / size_val, val_loss / size_val))
          
                train_losses.append(train_loss / size_train)
                val_losses.append(val_loss / size_val)
                train_accuracies.append(train_accuracy / size_train)
                val_accuracies.append(val_accuracy / size_val)
        
        
            # Save the variables to disk after training completed
            saver = tf.train.Saver(max_to_keep=0)
           # path = os.path.join(sys.path[0], 'Models\cnn-version')
            save_path = saver.save(sess, "C:\\youtubeproject\\Models\\cnn-version", global_step=0)
            print("Model saved in path: %s" % save_path)

        return train_losses, train_accuracies, val_losses, val_accuracies
    
    
    
    def predict(self, image_path):
        # get directory of this script to load pretrained model
        script_path = os.getcwd()
        #print("script path:",sys.path[0])
        # read given image
        directory, filename = os.path.split(image_path)
        os.chdir( directory )
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image_resized = tf.image.resize_images(image_decoded, [self.n_input,self.n_input])
        image_normalized = tf.image.per_image_standardization(image_resized)
        
        # convert image to tensorflow dataset
        data = tf.data.Dataset.from_tensors(image_normalized).batch(1)
        iterator = tf.data.Iterator.from_structure(data.output_types, data.output_shapes)
        data_X = iterator.get_next()
        data_init = iterator.make_initializer(data)
        os.chdir( script_path )
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            # restoring the trained model
            saver.restore(sess, './Models/cnn-version-0')
            os.chdir( directory )
            sess.run(data_init)
            try:
                while True:
                    # make prediction
                    pred = self.conv_net(data_X, False)
                    output = sess.run(pred)
            except tf.errors.OutOfRangeError:
                pass
        os.chdir(script_path)    
        return output
    


