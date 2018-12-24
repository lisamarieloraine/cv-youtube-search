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
import json
import time
import itertools

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

class Data:
    def __init__(self, annotation_dir, image_dir):
        """
        Modified version of the COCO API.
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        self.ann_dir = annotation_dir
        self.img_dir = image_dir
        # load dataset
        os.chdir( self.ann_dir )
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        print('loading annotations into memory...')
        tic = time.time()
        dataset = json.load(open('instances_train2017.json', 'r'))
        assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
        print('Done (t={:0.2f}s)'.format(time.time()- tic))
        self.dataset = dataset
        self.createIndex()
            
            
    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                anns[ann['id']] = ann
    
        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img
    
        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

    
        print('index created!')
    
        # create class members
        self.anns = anns
        self.imgs = imgs
        self.cats = cats
        
        print(self.anns[344065])
        print('\n')
        print(self.imgs[344065])
        print('\n')
        print(self.cats[52])
        
        
    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Taken from Microsoft COCO API.
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds)  == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]
        if not iscrowd == None:
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:
            ids = [ann['id'] for ann in anns]
        return ids


ann_path = "D:\\cocoapi\\annotations\\"
img_path = "D:\\cocoapi\\images\\"
data = Data(ann_path, img_path)
