'''
Utility functions
'''

import numpy as np

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
