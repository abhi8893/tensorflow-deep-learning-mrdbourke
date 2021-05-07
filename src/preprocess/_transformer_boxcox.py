'''
BoxCox Transformer
'''
import scipy
from sklearn.base import BaseEstimator, TransformerMixin

class BoxCoxTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, lmbda=None, alpha=None):
        self.lmbda = lmbda
        self.alpha = alpha
        
    def fit(self, X):
        _, self.lmbda_, self.confint_ = self.boxcox(X, self.lmbda, self.alpha)
        return self
    
    def transform(self, X):
        return scipy.special.boxcox(X, self.lmbda_)
    
    def inverse_transform(self, X):
        return scipy.special.inv_boxcox(X, self.lmbda_)
    
    @staticmethod
    def boxcox(X, lmbda, alpha):
        d = {'transformed': None, 'lmbda': lmbda, 'confint': None}
        res = scipy.stats.boxcox(X, lmbda, alpha)
        
        if lmbda is None:
            d['lmbda'] = res[1]
            
        if alpha is not None:
            d['confint'] =  res[-1]
            
        if isinstance(res, list) or isinstance(res, tuple):
            d['transformed'] = res[0]
        else:
            d['transformed'] = res
            
            
        return d.values()
        