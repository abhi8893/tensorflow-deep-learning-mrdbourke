'''
Helper functions
'''

def describe_tensor(tensor):
    '''
    Describe a tensor (print out useful info)
    '''
    print('Datatype:', tensor.dtype)
    print('Number of dimensions:', tensor.ndim)
    print('Shape of tensor:', tensor.shape)
    print('Elements along the 0 axis:',tensor.shape[0])
    print('Elements along the last axis:', tensor.shape[-1])
    print('Total number of elements:', tf.size(tensor).numpy())