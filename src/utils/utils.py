'''
Utility functions
'''
import tensorflow as tf


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

                                     
            
        