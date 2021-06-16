'''
Utility functions
'''
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import datetime
from pathlib import Path


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


def tensor_variance(tensor):
    mu = tf.reduce_mean(tensor)
    var = tf.reduce_sum((tensor - mu)**2)/tf.shape(tensor).numpy()
    return tf.squeeze(var)


def check_tfmodel_weights_equality(model1, model2):
    for w1, w2 in zip(model1.weights, model2.weights):
        res = tf.reduce_all(w1 == w2).numpy()
        if not res:
            return False 
    return True

def create_tensorboard_callback(experiment, task=None, parent_dir=None):

    if task is None:
        task = ''
    
    if parent_dir is None:
        parent_dir = ''
    
    log_dir = os.path.join(parent_dir, task, experiment, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.abspath(log_dir), profile_batch=0)
    print(f'Saving TensorBoard log files to " {log_dir}"')
    return tensorboard_callback
    

def get_series_group_counts(ser):
    counts = ser.value_counts()
    counts_perc = counts/ser.dropna().shape[0]
    count_df = pd.concat([counts, counts_perc], axis=1)
    count_df.columns = ['count', 'prop']
    
    return count_df


def reshape_classification_prediction(pred):
    if len(pred.shape) == 1:
        return pred[:, np.newaxis]
    elif pred.shape[-1] == 1:
        return pred
    elif pred.shape[-1] == 2:
        return pred[:, 1, np.newaxis]
    else:
        return pred


                                     
            
        