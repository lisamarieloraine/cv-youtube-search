# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 20:37:54 2018

@author: plagl
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import skimage
import coco

#ROOT_PATH = "D:\cocoapi\images"
#train_data_directory = os.path.join(ROOT_PATH, "train2017")


def set_up(path, annotation_file):
    # sets up the coco API
    os.chdir( path ) # change the directory to the directory containing the data
    retval = os.getcwd()
    print("Directory changed successfully:", retval)
    API = coco.COCO("instances_train2017.json")
    return API


def get_ids(cat_name, API):
    # gets a list of image ids containing a certain food object
    catId = API.getCatIds([cat_name])
    print("The category id for a", cat_name, "object is:", catId[0])
    imageIds = API.getImgIds(catIds=catId)
    return imageIds
    

def load_images(ids, API):
    # loads all images - WIP

    
    images = API.loadImgs(ids)

    return images


path = "D:\\cocoapi\\annotations\\"
annotation_file = "instances_train2017.json"
API = set_up(path, annotation_file)
imageIds = get_ids('apple', API)
print(imageIds[0:10]) # print sample image ids of the given type
images = load_images(imageIds, API)
print(type(images[0]))