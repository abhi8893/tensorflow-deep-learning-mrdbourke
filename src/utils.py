'''
Utility functions
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tempfile import TemporaryFile

def describe_tensor(tensor):
    '''Describe a tensor (print out useful info)'''
    
    print('Datatype:', tensor.dtype)
    print('Number of dimensions:', tensor.ndim)
    print('Shape of tensor:', tensor.shape)
    print('Elements along the 0 axis:',tensor.shape[0])
    print('Elements along the last axis:', tensor.shape[-1])
    print('Total number of elements:', tf.size(tensor).numpy())



def get_dataframe_cols(df, cols):
    '''Get dataframe columns which are present in the pandas dataframe'''
    
    cols = df.columns[df.columns.isin(cols)]
    
    return df[cols]


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))



class LabelAnalyzer:

    """Analyze Labels for classification problem.

        Args:
            train_labels ([type]): train labels.
            test_labels ([type], optional): test labels. Defaults to None.
            class_names ([type], optional): Actual class names or a mapping from class_values to class_labels. Defaults to None.
    """
    
    def __init__(self, train_labels, test_labels=None, class_names=None):


        self.train = self._get_count_dict(train_labels)
        self.__has_test_data = test_labels is not None

        class_values = self.train['value']
        
        if self.__has_test_data:
            self.test = self._get_count_dict(test_labels)
            class_values = tuple(set(class_values + self.test['value']))

        else:
            self.test = None


        if class_names is not None:
            if isinstance(class_names, (tuple, list, np.ndarray)):
                assert len(class_names) == len(class_values)
                class_labels = tuple(class_names)

            elif isinstance(class_names, dict):
                assert set(class_names.keys()).intersection(class_values) == set(class_values)

                class_labels = tuple(class_names.values())
                class_values = tuple(class_names.keys())

        else:
            class_labels = class_values


        self.class_labels = class_labels
        self.class_values = class_values
        self.n_classes = len(self.class_labels)
        self.val2lab = dict(zip(self.class_values, self.class_labels))
        self.lab2val = {lab: val for val, lab in self.val2lab.items()}
    

        self._make_count_df()
    
    @staticmethod
    def _get_count_dict(labels):
        unique, counts = np.unique(labels, return_counts=True)
        unique, counts = tuple(unique), tuple(counts)

        return dict(zip(['value', 'count'], [unique, counts]))


        
    def count(self, label, subset='train'):
        d = self.__getattribute__(subset)
        idx = d['value'].index(label)
        return d['count'][idx]
    
    def _make_count_df(self):
        traindf = pd.DataFrame(self.train)

        if not self.__has_test_data:
            countdf = traindf
        else:
            testdf = pd.DataFrame(self.test)
            countdf = pd.merge(traindf, testdf, how='outer', on='value', 
                            suffixes=('_train', '_test')).fillna(0).sort_values('value')
            
            countdf[['count_train', 'count_test']] = countdf[['count_train', 'count_test']].astype('Int64')
        
            countdf = countdf


        countdf = countdf.set_index('value').reindex(self.class_values).reset_index().fillna(0)
        countdf['label'] = countdf['value'].map(self.val2lab)

        first_cols = ['value', 'label']
        other_cols = list(countdf.columns[~countdf.columns.isin(first_cols)])
        
        self.countdf = countdf[first_cols + other_cols]

        
        
    def plot(self):

        ax = self.countdf.drop('value', axis=1).plot(x='label', kind='bar', stacked=True, figsize=(12, 4))

        if self.__has_test_data:
            title = 'Train vs Test label distribution'
            legend_labels = ['train', 'test']
        else:
            title = 'Train label distribution'
            legend_labels = ['train']

        ax.set_title(title, fontdict=dict(weight='bold', size=15))
        ax.legend(labels=legend_labels, bbox_to_anchor=(1.01, 0.6))
        plt.xticks(rotation=45)
        return ax


def plot_keras_model(model, to_file=None, show_shapes=False, 
                     show_dtype=False, show_layer_names=True, rankdir='TB', 
                     expand_nested=False, dpi=96):
    if to_file is None:
        t = TemporaryFile(suffix='.png')
        t.close()
        fname = t.name
    else:
        fname = to_file
        
    
    return tf.keras.utils.plot_model(model, to_file=fname, show_shapes=show_shapes, 
                                     show_dtype=show_dtype, show_layer_names=show_layer_names, 
                                     rankdir=rankdir, 
                                     expand_nested=expand_nested, dpi=dpi)
                                     
            
        