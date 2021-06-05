import tensorflow as tf
from sklearn import metrics as skmetrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ClassificationPerformanceComparer:
    
    MODEL_NON_UNIQUE_MSG = 'The model names are not unique! Please make model names unique or provide a dictionary of models'
    
    def __init__(self, models, data, class_names=None):
        
        if isinstance(models, (tuple, list)):
            self.models = {model.name: model for model in models}
            
            # TODO: which model name is duplicated? Show in message.
            assert len(self.models) == len(models), self.MODEL_NON_UNIQUE_MSG
            
        else:
            self.models = models
        
        if (class_names is None) and isinstance(data, tf.keras.preprocessing.image.DirectoryIterator):
            self.class_names = data.class_indices
        else:
            self.class_names = class_names
            
        self.data = data
        
    @staticmethod
    def _get_prediction_from_data(model, data):    
        if isinstance(data, tf.keras.preprocessing.image.DirectoryIterator):
            y_pred_prob = model.predict(data)
            y = data.labels
        elif isinstance(data, (tuple, list)):
            y_pred_prob = model.predict(data[0])
            y = data[1]
            
        return y, y_pred_prob.argmax(axis=1)
            
    def calculate_metric_comparison_df(self):

        self.predictions = {}
        
        compdf = []
        for name, model in self.models.items():
            y, y_pred = self._get_prediction_from_data(model, self.data)
            self.predictions[name] = y_pred
            crdf = pd.DataFrame(skmetrics.classification_report(y, y_pred, target_names=self.class_names, output_dict=True))
            crdf['model'] = name
            compdf.append(crdf)
            
            
        compdf = pd.concat(compdf)
        compdf.index.name = 'metric'
        compdf.reset_index(inplace=True)
        
        compdf_small = compdf.loc[compdf['metric'] != 'support', ['metric', 'weighted avg', 'model']].rename(columns={'weighted avg': 'value'})
        acc_df = compdf.loc[compdf['metric'] != 'support', ['model', 'accuracy']].drop_duplicates().melt(id_vars='model', var_name='metric')
        
        compdf_small = pd.concat([compdf_small, acc_df]).sort_values('model')
        
        
        self.compdf = compdf
        self.compdf_small = compdf_small
        
    
    
    def plot_metric_comparison_df(self):
        plt.figure(figsize=(8, 4))
        sns.barplot(x='metric', y='value', hue='model', data=self.compdf_small)
        plt.legend(bbox_to_anchor=[1.01, 0.6])
    
    
    
