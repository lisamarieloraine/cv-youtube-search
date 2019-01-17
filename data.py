import os
import coco
import json
import tensorflow as tf
from PIL import Image  # uses pillow

class Data:
    def __init__(self, annotation_dir,klasses,img_path_train, img_path_val ):
        
        self.ann_dir = annotation_dir
        self.klasses = klasses
        
        # load dataset
        os.chdir( self.ann_dir )
        self.API = coco.COCO('instances_train2017.json')
        self.img_dir = img_path_train
        self.data_train = self.select_images(klasses,581929)
        self.img_dir = img_path_val
        os.chdir( self.ann_dir )
        self.API = coco.COCO('instances_val2017.json')
        self.data_val = self.select_images(klasses,198806)

    def write_data(self):
        os.chdir(self.ann_dir)
        
        with open('data_train.txt', 'w') as filehandle:  
            json.dump(self.data_train, filehandle)
        
        with open('data_val.txt', 'w') as filehandle:  
            json.dump(self.data_val, filehandle)

    def get_cat(self,catnms):
        return self.API.getCatIds(catNms=catnms)
    
    def get_ids(self,catNms):
        return self.API.getImgIds(catIds = catNms)
    
    def get_imgs(self,ids):
        return self.API.loadImgs(ids)

    def prominent(self,imgids,catids):  #check if our object is prominent enough in the picture
        annIds = self.API.getAnnIds(imgIds=imgids,catIds=catids,iscrowd=False)
        ann = self.API.loadAnns(annIds)
        area = 0
        for item in ann:
            area += item['area']
        os.chdir( self.img_dir )
        image_name = self.get_imgs(imgids)[0]['file_name']
        im = Image.open(image_name)
        size_image = im.size[0]*im.size[1]
        size = size_image * 0.15

        if area < size:
            return False
        else:
            return True
        
    def select_images(self,klasses,limit_id):
        data_images = []
        for klasse in klasses:
            cat_lst = self.get_cat(klasse[0])
            id_lst = self.get_ids(cat_lst)
            imgs = self.get_imgs(id_lst)
            
            for item in imgs:
                if item['id'] < limit_id and self.prominent(item['id'],cat_lst):
                    data_images.append((item['file_name'],klasse[1]))
        return data_images
            

    
def create_dataset(data, batch_size,img_dir,res):#files is list of filenames, labels contains integer label in order
#def create_dataset(self, data_jpg, batch_size):
    # Reads an image from a file, decodes it into a dense tensor, and resizes it
    # to a fixed shape.
    def _parse_function(filename, label):
      image_string = tf.read_file(filename)
      image_decoded = tf.image.decode_jpeg(image_string, channels=3)
      image_resized = tf.image.resize_images(image_decoded, [res,res])
      image_normalized = tf.image.per_image_standardization(image_resized)
      return image_normalized, label
  
    def split_filenames(data_non_splitted):
        jpg_list = []
        labels_list = []
        for element in data_non_splitted:
            jpg_list.append(element[0])
            labels_list.append(element[1])
        return jpg_list,labels_list

    file_names,labels = split_filenames(data)

   #ValueError: Shape must be rank 0 but is rank 1 for 'ReadFile' (op: 'ReadFile') with input shapes: [?].
    # A vector of filenames.
    
    filenames = tf.constant(file_names)

    # `labels[i] is the label for the image in filenames[i].
    classes = tf.constant(labels)
    
    #might improve efficiency for large datasets
    #placeholder_X = tf.placeholder(filenames_train.dtype, filenames_train.shape)
    #placeholder_y = tf.placeholder(labels_train.dtype, labels_train.shape)
    
    # Create separate Datasets for training and validation
    os.chdir( img_dir )
    dataset = tf.data.Dataset.from_tensor_slices((filenames, classes))
    dataset = dataset.map(_parse_function).batch(batch_size)
    return dataset,labels  
            
            
            
            