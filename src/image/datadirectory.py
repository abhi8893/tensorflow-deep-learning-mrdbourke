import os
import glob
import pandas as pd
import numpy as np
from ..utils import LabelAnalyzer
from .dataset import ImageDataset
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import math
from collections import Counter

class ClassicImageDataDirectory:
    
    def __init__(self, data_dir, target_image_size, dtype=np.float32):
        self._verify_train_test_structure(data_dir)
        
        self.data_dir = data_dir
        self.target_image_size = target_image_size
        
        self.train = self.__get_subset_info('train')
        self.test = self.__get_subset_info('test')
        
        class_names = sorted(list(set(self.train['name'] + self.test['name'])))
        self.labelencoder = LabelEncoder()
        self.labelencoder.fit(class_names)
        
        self.train['label'] = list(self.labelencoder.transform(self.train['name']))
        self.test['label'] = list(self.labelencoder.transform(self.test['name']))
        
        train_dict = self.train.copy()
        train_dict.pop('name')
        train_dict.pop('dir')
        
        test_dict = self.test.copy()
        test_dict.pop('name')
        test_dict.pop('dir')
        
        la = LabelAnalyzer.from_countdict(train_dict, test_dict, class_names)
        ImageDataset._ImageDataset__set_attrs_from_labelanalyzer(self, la)

        self.dtype = dtype
                
        
    def load(self, batch_size=32, subset='train', class_names='all', dtype=None):
        if class_names == 'all':
            class_names = self.class_names

        if dtype is None:
            dtype = self.dtype
            
        subset_dir = os.path.join(self.__getattribute__(subset)['dir'])
        
        all_files = {nm: [] for nm in class_names}
        
        for nm in class_names:
            all_files[nm] = []
            try:
                all_files[nm] +=  self.list_data_files(subset, nm)
            except FileNotFoundError:
                pass
            
            
        cls_size = {nm: len(files) for nm, files in all_files.items()}
        total_size = sum(cls_size.values())
        cls_size_prop = {nm: size/total_size for nm, size in cls_size.items()}
        cls_batch_size = Counter(np.random.choice(self.class_names, size=batch_size, 
                                p=[cls_size_prop[nm] for nm in self.class_names]))

        
        total_batches = math.floor(total_size/batch_size) + 1
        
        for batch_num in range(total_batches):
            batch_files = []
            batch_labels = []
            for cls_name, files in all_files.items():
                start = batch_num*cls_batch_size[cls_name]
                end = start + cls_batch_size[cls_name]

                if start > cls_size[cls_name] or end > cls_size[cls_name]:
                    continue

                
                
                batch_files += [os.path.join(subset_dir, cls_name, f) for f in files[start:end]]
                batch_labels += [cls_name]*(len(files[start:end]))
                
            batch_images = np.array([self._load_and_resize_image(file, target_size=self.target_image_size, dtype=dtype)
                                     for file in batch_files])
            
            batch_labels = self.labelencoder.transform(batch_labels)
                
            yield ImageDataset((batch_images, batch_labels), classes=self.lab2nm)


    @staticmethod    
    def _load_and_resize_image(file, target_size, dtype):
        img = tf.keras.preprocessing.image.load_img(file, target_size=target_size)
        img = tf.keras.preprocessing.image.img_to_array(img, dtype=dtype)
        return img
        
    
    # TODO
    @staticmethod
    def _verify_train_test_structure(data_dir):
        pass
    
    @staticmethod
    def __list_files(folder):
        return [f for f in os.listdir(folder) if not f.startswith('.')]
    
    def list_data_files(self, subset, class_name):
        folder = os.path.join(self.data_dir, subset, class_name)
        return self.__list_files(folder)
    
    def __get_subset_info(self, subset):
        d = {}
        d['dir'] = os.path.join(self.data_dir, subset)
        d['name'] = self.__list_files(d['dir'])
        d['label'] = self.__list_files(d['dir'])
        d['count'] = [len(self.list_data_files(subset, nm)) for nm in d['name']]
        
        return d