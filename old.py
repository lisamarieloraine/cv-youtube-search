# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 13:50:16 2018

@author: plagl
"""

def set_up(path, annotation_file):
    # sets up the coco API
    os.chdir( path ) # change the directory to the directory containing the data
    retval = os.getcwd()
    print("Directory changed successfully:", retval)
    API = coco.COCO("instances_train2017.json")
    return API


def get_ids(cat_name, API):
    # gets a list of image ids containing a certain food object
    catId = API.getCatIds(catNms=[cat_name])
    print("The category id for a", cat_name, "object is:", catId[0])
    imageIds = API.getImgIds(catIds=catId)
    return imageIds

def get_dict(annotation_directory, annotation_file):
        os.chdir( annotation_directory )
        # load dataset
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            self.createIndex()
    

def load_images(ids, API, image_directory):
    # loads all images - WIP    
    images = []
    labels = []
    info = []
    
    apples = get_ids('apple', API)
    bananas = get_ids('banana', API)
    broccolis = get_ids('broccoli', API)
    
    info += API.loadImgs(ids)
    os.chdir( image_directory )
    file_names = [f.get('file_name') for f in info]
    
    for f in file_names:
        images.append(tf.image.decode_jpeg(f, channels=1))
        #labels.append(int(d))
        
    return images, info, labels
