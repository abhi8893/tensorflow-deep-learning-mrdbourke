import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from ..utils import LabelAnalyzer
import pandas as pd
import tensorflow as tf
import numpy as np


# TODO: Implement .copy method (https://stackoverflow.com/questions/45051720/best-practice-to-implement-copy-method/45094347)
class ImageDataset:
    
    def __init__(self, train_data, test_data=None, classes=None):
        self.train_images, self.train_labels = [x.copy() for x in train_data]
        self.__has_test_data = test_data is not None
        
        if test_data is None:
            self.test_images, self.test_labels = (None, None)
        else:
            self.test_images, self.test_labels = [x.copy() for x in test_data]

        
        la = LabelAnalyzer(self.train_labels, self.test_labels, classes)
        self.__set_attrs_from_labelanalyzer(la)
        
        self.train_size, *self.train_dim = self.train_images.shape
        
        if self.__has_test_data:
            self.test_size, *self.test_dim = self.test_images.shape


    def __len__(self):
        return len(self.train_images)
    
    def __set_attrs_from_labelanalyzer(self, la):
        self.class_labels = la.class_labels
        self.class_names = la.class_names
        self.n_classes = la.n_classes
        self.lab2nm = la.lab2nm
        self.nm2lab = la.nm2lab
        self.labelcountdf = la.countdf
        self.plot_labelcounts = la.plot
        
        
    def __check_test_subset(self, subset):
        if subset == 'test' and not self.__has_test_data:
            raise ValueError('Test data is not provided!')
        
        
    def view_random_images(self, class_names='any', n_each=1, subset='train', cmap='auto'):
        
        self.__check_test_subset(subset)
        
        subset_labels = self.__getattribute__(f'{subset}_labels')
        subset_images = self.__getattribute__(f'{subset}_images')
        
        
        if cmap == 'auto':
            subset_dim = len(subset_images[0, :].shape)
            if (subset_dim == 2) or (subset_dim == 3 and subset_images.shape[-1] == 1): 
                cmap = plt.cm.binary
            else:
                cmap = None
        
        
        if class_names is 'any':
            class_names = [np.random.choice(self.class_names)]
        elif class_names is 'all':
            class_names = self.class_names
        
        n_classes = len(class_names)

        fig  = plt.figure(figsize=(n_classes*2, n_each*2))
        gs = GridSpec(n_each, n_classes)
        gs.update(wspace=0.15, hspace=0.05)
        
        # Get class labels
        class_labels = [self.nm2lab[nm] for nm in class_names]
        
        # Get label indices for each class
        label_df = pd.DataFrame(subset_labels)
        label_indices = label_df.groupby([0]).indices
        
        # Now sample from chosen class_labels
        sample_labels = {k: np.random.choice(label_indices[k], n_each) for k in class_labels}
        
            
        for i, (cls_lab, indices) in enumerate(sample_labels.items()):
            for j, idx in enumerate(indices):
                
                ax = plt.subplot(gs[j, i])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                cls_nm = self.lab2nm[cls_lab]
                img = subset_images[idx]
                ax.imshow(img, cmap=cmap)
                ax.set_axis_off()
                ax.set_aspect('auto')
                
                if j == 0:
                    ax.set_title(cls_nm, fontdict=dict(weight='bold'))
                    
        return np.reshape(fig.axes, (n_each, len(class_labels)))
    
    
    def select(self, class_names, apply_on_subset='all'):
        assert apply_on_subset in ['train', 'test', 'all']
        
        if apply_on_subset == 'all':
            chosen_subsets = ['train']
            if self.__has_test_data:
                chosen_subsets.append('test')
        else:
            chosen_subsets = [apply_on_subset] 
            
        kwarg_dict = {'classes': self.lab2nm}
    
        for subset in chosen_subsets:  
            subset_labels = self.__getattribute__(f'{subset}_labels')
            subset_images = self.__getattribute__(f'{subset}_images')

            class_labels = [self.nm2lab[nm] for nm in class_names]
            idx = np.isin(subset_labels, class_labels)
            kwarg_dict[f'{subset}_data'] = (subset_images[idx], subset_labels[idx])
            
            
        for subset in ['train', 'test']:
            if not kwarg_dict.get(f'{subset}_data'):
                subset_labels = self.__getattribute__(f'{subset}_labels')
                subset_images = self.__getattribute__(f'{subset}_images')

                if (subset_labels is None) and (subset_images is None):
                    kwarg_dict[f'{subset}_data'] = None
                else:
                    kwarg_dict[f'{subset}_data'] = (subset_images, subset_labels)
     
        imgds = self.__class__(**kwarg_dict)
        
        return imgds
          
        
    def sample(self, n):
        pass
    
    
    def get_one_hot_labels(self, subset):
        subset_labels = self.__getattribute__(f'{subset}_labels')
        return tf.keras.utils.to_categorical(subset_labels, num_classes=self.n_classes)


    def __getitem__(self, idx):

        if isinstance(idx, int):
            idx = slice(idx, idx+1)

        train_data = (self.train_images[idx], self.train_labels[idx])
        if self.__has_test_data:
            test_data = (self.test_images[idx], self.test_labels[idx])
        else:
            test_data = None


        imgds = self.__class__(train_data, test_data, classes=self.lab2nm)

        return imgds
        