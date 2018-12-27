# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 13:39:00 2018

@author: plagl
"""

# Import modules
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets


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















# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string)
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label

# A vector of filenames.
filenames = tf.constant(["/var/data/image1.jpg", "/var/data/image2.jpg", ...])

# `labels[i]` is the label for the image in `filenames[i].
labels = tf.constant([0, 37, ...])

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(_parse_function)



# Create some variables.
ann_path = "D:\\cocoapi\\annotations\\"
img_path = "D:\\cocoapi\\images\\"
data = Data(ann_path, img_path)
anns = data.get_annotations()
info = data.get_info(anns)
lbl = data.load_labels(anns)
classes, counts = data.convert_labels(lbl)
# img = data.load_images(info)



# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, and save the
# variables to disk.
with tf.Session() as sess:
  sess.run(init_op)
  # Do some work with the model.
  # Save the variables to disk.
  save_path = saver.save(sess, "D:\\cocoapi\\tmp\\images.ckpt")
  print("Model saved in path: %s" % save_path)