import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_STORE = {
    'X1':lambda X: X[:, 0],
    'X2': lambda X: X[:, 1],
    'X1_sq': lambda X: X[:, 0]**2,
    'X2_sq': lambda X: X[:, 1]**2,
    'X1X2': lambda X: X[:, 0]*X[:, 1],
    'sin(X1)': lambda X: np.sin(X[:, 0]),
    'sin(X2)': lambda X: np.sin(X[:, 1])
    }

class TfPlayDataset:
    
    def __init__(self, X: np.array, y: np.array, features=['X1', 'X2'], 
                 scale=True, train_test_ratio=0.5, random_state=None):
        assert X.shape[1] == 2
        assert X.shape[0] == y.shape[0]
        
        self.features = features      
        self.__data = self._make_dataframe(X, y)
        
        idx = {}
        idx['train'], idx['test'] = train_test_split(self.__data.index, test_size=1/(1+train_test_ratio))
        
        
        # TODO: Make below code DRY
        
        self.train = {}
        self.train['data'] = self.__data.loc[idx['train'], ['X1', 'X2']].copy()
      
        
        self.test = {}
        self.test['data'] = self.__data.loc[idx['test'], ['X1', 'X2']].copy()
        
        if scale:
            self.scaler = StandardScaler()
            self.__data.loc[idx['train'], features] = self.scaler.fit_transform(self.__data.loc[idx['train'], features])
            self.__data.loc[idx['test'], features] = self.scaler.transform(self.__data.loc[idx['test'], features])
            
            
        self.train['features'] = self.__data.loc[idx['train'], features]
        self.train['label'] = self.__data.loc[idx['train'], ['label']]
        
        self.test['features'] = self.__data.loc[idx['test'], features]
        self.test['label'] = self.__data.loc[idx['test'], ['label']]
        
        
    def _make_dataframe(self, X, y):
        df = pd.DataFrame(X, columns=['X1', 'X2'])
        features = [f for f in self.features if f not in ['X1', 'X2']]
        df = pd.concat([df, self._featurize(X, features)], axis=1)
        df['label'] = y
        
        return df
        
    
    @staticmethod
    def _featurize(X, features):
        features_df = pd.DataFrame({feat: FEATURE_STORE[feat](X) for feat in features})
        return features_df
    
    
    def plot(self, subset='train', ax=None):
        
        if ax is None:
            fig, ax = plt.subplots()
        
        
        subset_dict = self.__getattribute__(subset)
        
        X = subset_dict['data'].values
        y = subset_dict['label'].values.flatten()

        ax.scatter(X[y == 0, 0], X[y == 0, 1], color='red', label='0')
        ax.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='1')
        ax.set(xlabel='X1', ylabel='X2')
        
        plt.legend()
        
        return ax
        
        
        