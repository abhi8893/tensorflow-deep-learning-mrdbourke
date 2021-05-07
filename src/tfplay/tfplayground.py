from tensorflow.keras import layers, losses, optimizers, regularizers
from ..visualize import plot_learning_curve, plot2d_decision_function, plot_confusion_matrix
from .tfplaydataset import TfPlayDataset
import sklearn.datasets as skdata
import tensorflow as tf


class TensorflowPlayground:
    
    
    def __init__(self, 
                 dataset: str,
                 train_test_ratio=0.5,
                 noise=0,
                 features=['X1', 'X2'],
                 scale=True,
                 n_samples=1000,
                 neurons=(4, ),
                 learning_rate=0.03,
                 activation='tanh',
                 regularization=None,
                 regularization_rate=0,
                 random_state=None):
        
        
        if dataset != 'circle':
            raise NotImplementedError("Only 'circle' dataset is implemented!")
            
        
        self.dataset = dataset
        self.train_test_ratio = train_test_ratio
        self.noise = noise
        self.features = features
        self.scale = scale
        self.n_samples = n_samples
        self.neurons = neurons
        self.learning_rate = learning_rate
        self.activation = activation
        self.regularization = regularization
        self.regularization_rate = regularization_rate
        self.random_state = random_state
            
        
        # Make dataset
        self.make_data()
        # Make model
        self.make_model()
         
        
    def make_data(self):
        if self.dataset == 'circle':
            X, y = self._make_circle_X_y(self.noise, self.n_samples, self.random_state)
        
        self.data = TfPlayDataset(X, y, self.features, self.scale, self.train_test_ratio, self.random_state)
            
    @staticmethod        
    def _make_circle_X_y(noise=0, n_samples=1000, random_state=None):
        return skdata.make_circles(n_samples=n_samples, noise=noise, random_state=random_state)

    
    @staticmethod
    def _make_model(input_shape, neurons, learning_rate, 
                   activation, regularization=None, regularization_rate=0):
        
        if regularization == 'L1':
            reg = regularizers.l1(regularization_rate)
        elif regularization == 'L2':
            reg = regularizers.l2(regularization_rate)
        else:
            reg = None
            
        model = tf.keras.models.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=input_shape))
        
        num_layers = len(neurons)
        
        # Hidden layers
        for i in range(num_layers):
            model.add(layers.Dense(neurons[i], activation=activation, kernel_regularizer=reg))
            
        # Output layer
        model.add(layers.Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(loss=losses.binary_crossentropy, optimizer=optimizers.Adam(learning_rate=learning_rate))
        
        
        return model
                
            
    def make_model(self):
        num_feats = len(self.features)
        self.model = self._make_model(num_feats, self.neurons, self.learning_rate, 
                                      self.activation, self.regularization, self.regularization_rate)
        
    def train(self, epochs=10, batch_size=None):
        self.model.fit(self.data.train['features'], self.data.train['label'], epochs=epochs, batch_size=batch_size)
        
        
    def predict(self, X):
        X_feat = self.data._featurize(X, self.features)
        if self.scale:
            X_feat = self.data.scaler.transform(X_feat)
            
        return self.model.predict(X_feat)
    
    def plot_learning_curve(self):
        return plot_learning_curve(self.model.history.history)
    
    
    def plot_decision_function(self, ax=None):
        subset_dict = self.data.train
        cp = plot2d_decision_function(self.predict, subset_dict['data'].values, ax=ax)
        return cp
    
    
    def plot_confusion_matrix(self, subset='test'):
        subset_dict = self.data.__getattribute__(subset)
        y_true = subset_dict['label']
        y_pred = self.predict(subset_dict['data'].values).round()
        return plot_confusion_matrix(y_true, y_pred)
    
    