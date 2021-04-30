import pandas as pd
import numpy as np
from .utils import get_dataframe_cols, rmse
import matplotlib.pyplot as plt
import tensorflow as tf



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


def plot1d_reg_preds(X: np.array, y: np.array, models, labels, figsize=(12, 7)):
    '''Models must be callable predict functions'''
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(X, y, s=0.2)
    for model, label in zip(models, labels):
        y_pred = np.squeeze(model(X))
        rmse_val = rmse(y, y_pred)
        ax.plot(X, y_pred, label=f'{label} (RMSE:{rmse_val: .2f})')
        
    plt.legend()
    
    return fig, ax


def plot2d_decision_function(model, X, ax=None):
    """Plot decision contours for a binary classification model.

    Args:
        model (callable): Prediction function which outputs probabilities.
        X (np.array): Input features of shape (None, 2)
        ax (matplotlib.axes._subplots.AxesSubplot, optional): A matplotlib axes object. Defaults to None.
    """


    x1_range = np.linspace(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1, 100)
    x2_range = np.linspace(X[:, 1].min() - 0.1, X[:, 1].max() + 0.1, 100)

    xx1, xx2 = np.meshgrid(x1_range, x2_range)
    X_grid = np.array((xx1.ravel(), xx2.ravel())).T
    y_grid_pred = model(X_grid)

    if ax is None:
        fig, ax = plt.subplots()

    color_levels = np.linspace(0, 1, 10).round(1)

    n1, n2 = len(x1_range), len(x2_range)
    grid = np.concatenate((X_grid, y_grid_pred), axis=1).reshape(n1, n2, 3)

    cp = ax.contourf(grid[:, :, 0], grid[:, :, 1], grid[:, :, 2], 
                     cmap='RdBu', levels=color_levels)

    return cp




        
    


        
    