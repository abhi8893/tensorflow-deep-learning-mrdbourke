import pandas as pd
import numpy as np
from .utils import get_dataframe_cols, rmse
import matplotlib.pyplot as plt



# TODO: Support for `extra_metrics` as a list
def plot_learning_curve(history_dict, extra_metric=None, include_validation=True):
    '''Plots the loss and the extra metric curve for train and val set'''
    
    if extra_metric is None:
        fig, axn = plt.subplots(1, 1)
        axn = np.array([axn])
    else:
        fig, axn = plt.subplots(2, 1, figsize=(8, 12))
        
    
    history_df = pd.DataFrame(history_dict)
    
    cols = ['loss']
    if include_validation:
        cols += ['val_loss']
    
    # Plot loss curve
    get_dataframe_cols(history_df, cols).plot(ax=axn[0])
    axn[0].set_title('loss', fontdict=dict(weight='bold', size=20))
    
    # Plot extra metric curve
    if extra_metric:
        cols = [extra_metric]
        if include_validation:
            cols += [f'val_{extra_metric}']
            
            
        get_dataframe_cols(history_df, cols).plot(ax=axn[1])
        axn[1].set_title(extra_metric, fontdict=dict(weight='bold', size=20))
        
    
    return fig, axn


def plot1d_reg_preds(X: np.array, y: np.array, models, labels):
    '''Models must be callable predict functions'''
    
    fig, ax = plt.subplots()
    ax.scatter(X, y, s=0.2)
    for model, label in zip(models, labels):
        y_pred = np.squeeze(model(X))
        rmse_val = rmse(y, y_pred)
        ax.plot(X, y_pred, label=f'{label} (RMSE:{rmse_val: .2f})')
        
    plt.legend()
    
    return fig, ax



        
    


        
    