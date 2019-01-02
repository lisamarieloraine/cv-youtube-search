# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 11:18:07 2018

@author: plagl
"""

import os
import coco
import json
import tensorflow as tf

class Data:
    def __init__(self, annotation_dir, annotation_name, image_dir):
        
        self.ann_dir = annotation_dir
        self.img_dir = image_dir
        
        # load dataset
        os.chdir( self.ann_dir )
        retval = os.getcwd()
        print("Directory changed successfully:", retval)
        self.API = coco.COCO(annotation_name)
        
        
    def get_annotations(self, crowds=0):
        ann_ids = self.API.getAnnIds(imgIds=[], catIds=[], areaRng=[], iscrowd=crowds)
        anns = self.API.loadAnns(ann_ids)
        return anns
    
    
    def get_info(self, anns):
        img_ids = [a.get('image_id') for a in anns]
        info = self.API.loadImgs(img_ids)
        return info
    
    
    def load_labels(self, anns): # ---------------- not in use anymore!!!
        print('creating labels...')
        labels = dict()
        for a in anns:
            labels[a.get('image_id')] = a.get('category_id')   
        return labels
            
    def convert_labels_old(self, labels): # ---------------- not in use anymore!!!
        apple = self.API.getCatIds(catNms=['apple'])[0]
        banana = self.API.getCatIds(catNms=['banana'])[0]
        broccoli = self.API.getCatIds(catNms=['broccoli'])[0]
        
        (other, apples, bananas, broccolis) = (0,1,2,3)
        counts = [0,0,0,0]
        
        for key, value in labels.items():
            if value == apple:
                labels[key] = apples
                counts[apples] += 1
            elif value == banana:
                labels[key] = bananas
                counts[bananas] += 1
            elif value == broccoli:
                labels[key] = broccolis
                counts[broccolis] += 1
            else:
                labels[key] = other
                counts[other] += 1
                
        # Save to json file
        with open('classes.json', 'w') as fp:
            json.dump(labels, fp)
        print('saved labels to json file!')       
        return labels, counts
    
    
    def convert_labels(self, anns):
        print('converting labels...') 
        apple = self.API.getCatIds(catNms=['apple'])[0]
        banana = self.API.getCatIds(catNms=['banana'])[0]
        broccoli = self.API.getCatIds(catNms=['broccoli'])[0]
        (other, apples, bananas, broccolis) = (-1,0,1,2)
        counts = [0,0,0,0]
        
        for a in anns:
            if a.get('category_id') == apple:
                a['category_id'] = apples
                counts[apples] += 1
            elif a.get('category_id') == banana:
                a['category_id'] = bananas
                counts[bananas] += 1
            elif a.get('category_id') == broccoli:
                a['category_id'] = broccolis
                counts[broccolis] += 1
            else:
                a['category_id'] = other
                counts[other] += 1
                
        # Save to json file
        with open('annotations.json', 'w') as fp:
            json.dump(anns, fp)
        print('saved annotations to json file!')       
        return anns, counts
    
    
    def load_images(self, info): # ---------------- not in use anymore!!!
        # converts images to tensors
        print('converting images to tensors...')
        images = dict()
        os.chdir( self.img_dir )
        for i in info:
            images[i.get('id')] = tf.image.decode_jpeg(i.get('file_name'), channels=1)
        return images
    
    
    # ---------------- not in use anymore!!!
    def create_dataset(self, info, anns):        
        # Reads an image from a file, decodes it into a dense tensor, and resizes it
        # to a fixed shape.
        def _parse_function(filename, label):
          image_string = tf.read_file(filename)
          image_decoded = tf.image.decode_jpeg(image_string)
          image_resized = tf.image.resize_images(image_decoded, [28, 28])
          return image_resized, label
        
        print('creating dataset...')
        os.chdir( self.img_dir )
        # A vector of filenames.
        filenames = tf.constant([f.get('file_name') for f in info])
        # `labels[i]` is the label for the image in `filenames[i].
        labels = tf.constant([f.get('category_id') for f in anns])
        
        # Apply one-hot encoding to labels
        # This converts the integer labels to a vector of 0s with a 1 for the 
        # category that the object belongs to
        #labels = tf.one_hot(labels, depth=3)
        
        #might improve efficiency for large datasets
        #filenames_placeholder = tf.placeholder(filenames.dtype, filenames.shape)
        #labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
        
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        dataset = dataset.map(_parse_function)
        print('dataset created!')
        return dataset
            

            
            
            
            
            
            