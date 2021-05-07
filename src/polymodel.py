'''
Polynomial model class
'''
import numpy as np

class PolynomialModel:
    
    
    def __init__(self, params, stderr):
        """Polynomial model class

        Args:
            params (list-like): Parameters of the polynomial (constant, 1st power, 2nd power ...)
            stderr (float): The standard deviation of the error
        """

        self.params = params
        self.stderr = stderr
        self.degree = len(self.params) - 1
        
    def __call__(self, X: np.array, add_noise=False):
        
        val = 0.0
        for i, p in enumerate(self.params):
            val += (X**i)*p
            
        val = val.sum(axis=1)
            
        if add_noise:
            val += self._sample_normal_noise(self.stderr, shape=(val.shape[0],))
            
        return val
    
            
    @staticmethod
    def _sample_normal_noise(stderr, shape=None):
        return np.random.normal(loc=0, scale=stderr, size=shape)
    
    
    def plot(self, X: np.array, ax=None, **kwargs):
        
        if ax is None:
            fig, ax = plt.subplots()
            
        y = self(X)
        
        ax.plot(X.flatten(), y.flatten(), **kwargs)
        
        return ax
    
    
    
        