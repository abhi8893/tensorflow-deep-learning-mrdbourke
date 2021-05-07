'''
OLS Regression model
'''
import numpy as np

class OLSRegressor:
    
    def __init__(self, degree=1, add_bias=True):
        
        if degree != 1:
            raise NotImplementedError('Providing a degree other than 1 is not implemented yet!')
        
        self._check_degree(degree)
        
        self.degree = degree
        self.add_bias = add_bias
        
        
    def fit(self, X, y):
        if self.add_bias:
            X = self._add_bias(X)
            
        params = np.linalg.inv(X.T@X)@X.T@y
        
        if self.add_bias:
            self.intercept_, self.coef_ = params[0:1], params[1:]
        else:
            self.intercept_, self.coef_ = np.array([0.0]), params
            
    def predict(self, X):
        beta_hat = np.expand_dims(np.concatenate([self.intercept_, self.coef_]), axis=-1)
        if self.add_bias:
            X = self.add_bias(X)
            
        return X@beta_hat
        
    @staticmethod
    def _add_bias(X):
        bias = np.ones(shape=(X.shape[0], 1))
        return np.hstack((bias, X))
        
    @staticmethod
    def _check_degree(degree):
        assert type(degree) is int, 'Please provided an integer value for degree'
        assert degree >= 0, 'Please provide a positive value for degree'
        
    