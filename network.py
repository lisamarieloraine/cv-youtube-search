# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 14:47:15 2018

@author: plagl
"""

# Import modules
import tensorflow as tf

# Initialize two constants
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# Multiply
result = tf.multiply(x1, x2)

# Intialize the Session
sess = tf.Session()

# Print the result
print(sess.run(result))

# Close the session
sess.close()



class CNN:
    def __init__(self, train_data, val_data, test_data, params):
        pass
    
    
    def train(self):
        pass
    
    
    def validate(self):
        pass
    
    
    def test(self):
        pass
    
    
    def predict(self):
        pass
