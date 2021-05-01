'''
Utility functions
'''

import numpy as np
import pandas as pd

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
    
    def __init__(self, train_labels, test_labels=None):
        self.train = self._get_count_dict(train_labels)
        self.__has_test_data = test_labels is not None
        
        if self.__has_test_data:
            self.test = self._get_count_dict(test_labels)

        self._make_count_df()
    
    @staticmethod
    def _get_count_dict(labels):
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(['label', 'count'], [list(unique), list(counts)]))
        
    def count(self, label, subset='train'):
        d = self.__getattribute__(subset)
        idx = d['label'].index(label)
        return d['count'][idx]
    
    def _make_count_df(self):
        traindf = pd.DataFrame(self.train)

        if not self.__has_test_data:
            self.countdf = traindf
            return None

        testdf = pd.DataFrame(self.test)
        countdf = pd.merge(traindf, testdf, how='outer', on='label', 
                           suffixes=('_train', '_test')).fillna(0).sort_values('label')
        
        countdf[['count_train', 'count_test']] = countdf[['count_train', 'count_test']].astype('Int64')
        
        self.countdf = countdf
        
        
    def plot(self):
        ax = self.countdf.plot(x='label', kind='bar', stacked=True, figsize=(12, 4))

        if self.__has_test_data:
            title = 'Train vs Test label distribution'
            legend_labels = ['train', 'test']
        else:
            title = 'Train label distribution'
            legend_labels = ['train']

        ax.set_title(title, fontdict=dict(weight='bold', size=15))
        ax.legend(labels=legend_labels, bbox_to_anchor=(1.01, 0.6))
        return ax


    
            
        
    
            
        