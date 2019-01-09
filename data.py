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
        
    
    def create_dict(self, file_name_info, file_name_anns):
        ann_ids = self.API.getAnnIds(imgIds=[], catIds=[], areaRng=[], iscrowd=False)
        anns = self.API.loadAnns(ann_ids)
        img_ids = [a.get('image_id') for a in anns]
        info = self.API.loadImgs(img_ids)
        images = dict()
        
        # getting categories and image ids of unique images
        for ann in anns:
            if 'category_id' in ann.keys() and ann.get('image_id') not in images.keys():
                images[ann.get('image_id')] = [ann.get('category_id')]       
        for i in info:
            if i.get('file_name') not in images[i.get('id')]:
                images[i.get('id')].append(i.get('file_name'))       

        return images
    
    
    def convert_dict(self, images, file_name):
        print('converting labels...') 
        apple = self.API.getCatIds(catNms=['apple'])[0]
        banana = self.API.getCatIds(catNms=['banana'])[0]
        broccoli = self.API.getCatIds(catNms=['broccoli'])[0]
        (other, apples, bananas, broccolis) = (-1,0,1,2)
        counts = [0,0,0,0]
        
        for key, value in images.items():
            if not len(value) == 2:
                print('Warning: not all info available for image', key)
            if not type(value[0]) == int:
                print('Warning: category is of wrong type for image', key)
            if not type(value[1]) == str:
                print('Warning: file name is of wrong type for image', key)
                
            if value[0] == apple:
                value[0] = apples
                counts[1] += 1
            elif value[0] == banana:
                value[0] = bananas
                counts[2] += 1
            elif value[0] == broccoli:
                value[0] = broccolis
                counts[3] += 1
            else:
                value[0] = other
                counts[0] += 1
                     
        # Save to json file
        with open(file_name, 'w') as fp:
            json.dump(images, fp)
        print('saved annotations to json file:', file_name)  
        print(counts)
        return images, counts
    
    

    def subset_images(self, images, counts):
        max_examples = max(counts[1:])
        labels = [i[0] for i in images.values()]
        file_names = [i[1] for i in images.values()]
        to_remove = counts[0] - max_examples
        remove_count = 0
        
        subset_labels = []
        subset_files = []

        for index, label in enumerate(labels):
            if label == -1 and remove_count < to_remove:
                remove_count += 1
            else:
                subset_labels.append(labels[index])
                subset_files.append(file_names[index])
    
        print("Number of examples BEFORE subsetting:", len(labels))
        print("Number of examples AFTER subsetting:", len(subset_labels))

        # Save to json files
        #with open(file_name_ann, 'w') as fp:
            #json.dump(labels, fp)
        #with open(file_name_info, 'w') as fp:
            #json.dump(file_names, fp)
        return subset_labels, subset_files
            
         
    def create_dataset(self, files, labels, batch_size):        
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
        filenames = tf.constant(files)

        # `labels[i]` is the label for the image in `filenames[i].
        classes = tf.constant(labels)
        
        #might improve efficiency for large datasets
        #placeholder_X = tf.placeholder(filenames_train.dtype, filenames_train.shape)
        #placeholder_y = tf.placeholder(labels_train.dtype, labels_train.shape)
        
        # Create separate Datasets for training and validation
        os.chdir( self.img_dir )
        dataset = tf.data.Dataset.from_tensor_slices((filenames, classes))
        dataset = dataset.map(_parse_function).batch(batch_size)
        print('dataset created!')
        
        return dataset      
            
            
            
            
            