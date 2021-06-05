import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.layers.experimental.preprocessing as KerasPreprocessing
import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers
from ..evaluate import KerasMetrics
from ..utils import create_tensorboard_callback



class TfHubDirectoryNotSettedError(Exception):
    pass


class ImageClassificationExperiment:
        
    IMAGE_DIM = (224, 224)
    NUM_CHANNELS = 3
    INPUT_SHAPE = (*IMAGE_DIM, NUM_CHANNELS)
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2
    CLASSIFICATION_TYPE = 'categorical'
    SCALING_FACTOR = 1/255.
    SEED = 42
    DEFAULT_UNFREEZE_NUM_LAYERS = 10
    
    EXPERIMENT_COUNTER = 1
    
    def __init__(self, name=None, initial_mode='feature_extraction', tfhub_log_dir=None):
        
        self.mode = initial_mode
        self._assign_name(name)
        self.tfhub_log_dir = tfhub_log_dir
        self.training_history = []
        
        
    def _assign_name(self, name):
        exp_id = self.EXPERIMENT_COUNTER
        if name is None:
            name = f'experiment_{exp_id}'
        
        self.name = name
        self.__class__.EXPERIMENT_COUNTER += 1
        
    def preprocess_data(self, data_augment=True, scale=True):
        self.train_datagen = ImageDataGenerator(validation_split=self.VALIDATION_SPLIT)
        self.test_datagen = ImageDataGenerator()
        
        self.data_augment = data_augment
        self.scale = scale
        
    def setup_directories(self, data_dir):
        self.data_dir = data_dir
        self.train_dir = os.path.join(data_dir, 'train')
        self.test_dir = os.path.join(data_dir, 'test')
        
    def import_data_from_directories(self):
        self.train_data = self.train_datagen.flow_from_directory(directory=self.train_dir, target_size=self.IMAGE_DIM, 
                                                                 batch_size=self.BATCH_SIZE, class_mode=self.CLASSIFICATION_TYPE, 
                                                                 subset='training', seed=self.SEED)
        
        self.validation_data = self.train_datagen.flow_from_directory(directory=self.train_dir, target_size=self.IMAGE_DIM,
                                                                     batch_size=self.BATCH_SIZE, class_mode=self.CLASSIFICATION_TYPE,
                                                                     subset='validation', seed=self.SEED)
        
        self.test_data = self.test_datagen.flow_from_directory(directory=self.test_dir, target_size=self.IMAGE_DIM, 
                                                              batch_size=self.BATCH_SIZE, class_mode=self.CLASSIFICATION_TYPE,
                                                              shuffle=False)
        
        
        self.n_classes = self.train_data.num_classes
        
        
    def create_model(self, pretrained_model, downstream_model=None, add_output_layer=True):
        
        self.pretrained_model = pretrained_model
        
        inputs = layers.Input(shape=self.INPUT_SHAPE, name='input_layer')
        
        if self.data_augment:
            x = self._get_data_augment_layer()(inputs)
        else:
            x = inputs
            
        if self.scale:
            x = KerasPreprocessing.Rescaling(self.SCALING_FACTOR)(x)
            
        self.pretrained_model.trainable = False
        
        features = self.pretrained_model(x)
        
        if downstream_model is None:
            self.downstream_model = self._get_default_downstream_model()
        else:
            self.downstream_model = downstream_model
        
        final_hidden = self.downstream_model(features)
        
        if add_output_layer:
            outputs = layers.Dense(self.n_classes, activation='softmax', name='output_layer')(final_hidden)
        else:
            outputs = final_hidden
        
        self.model = tf.keras.models.Model(inputs, outputs)
        
        self.set_training_mode(self.mode)
        
    @staticmethod
    def _get_default_downstream_model():
        model = tf.keras.Sequential([
            layers.GlobalAvgPool2D(name='global_avg_pooling_layer')
        ], name='downstream_model')
        
        return model
            
    
    def compile_model(self, learning_rate=0.001):
        self.model.compile(loss=losses.categorical_crossentropy, optimizer=optimizers.Adam(learning_rate=learning_rate),
                           metrics=[KerasMetrics.f1, 'accuracy'])
        
    def _get_data_augment_layer(self):
        data_augment_layer = tf.keras.models.Sequential([
            KerasPreprocessing.RandomFlip('horizontal'),
            KerasPreprocessing.RandomRotation(0.2),
            KerasPreprocessing.RandomZoom(0.2),
            KerasPreprocessing.RandomHeight(0.2),
            KerasPreprocessing.Resizing(height=self.IMAGE_DIM[0], width=self.IMAGE_DIM[1])
        ])
        
        return data_augment_layer
    
    def set_training_mode(self, mode, unfreeze_last_n=None):
        if mode == 'feature_extraction':
            assert unfreeze_last_n is None, 'This argument is only valid for mode == "fine_tuning"'
            self.pretrained_model.trainable = False
        elif mode == 'fine_tuning':
            self.pretrained_model.trainable = True
            if unfreeze_last_n is None:
                unfreeze_last_n = self.DEFAULT_UNFREEZE_NUM_LAYERS
            for layer in self.pretrained_model.layers[:-unfreeze_last_n]:
                layer.trainable = False
                
        
        self.mode = mode
    
    def run(self, epochs=10, tfhub_log=False):
        
        if tfhub_log:
            if self.tfhub_log_dir is None:
                raise TfHubDirectoryNotSettedError('Set the directory to save tensorflow hub logs first!')
            
            log_dir = os.path.join(self.tfhub_log_dir, self.name, self.mode)
            model_callbacks = [create_tensorboard_callback(log_dir)]
        else:
            model_callbacks = []
            
        try:
            initial_epoch = self.model.history.epoch[-1]
        except (IndexError, AttributeError):
            initial_epoch = 0
            
            
        self.model.fit(self.train_data, steps_per_epoch=len(self.train_data),
                      validation_data=self.validation_data, validation_steps=len(self.validation_data),
                       epochs=epochs+initial_epoch, initial_epoch=initial_epoch, callbacks=model_callbacks)
        
        
        history_dict = self.model.history.history.copy()
        history_dict['epoch'] = self.model.history.epoch
        
        self.training_history.append((self.mode, history_dict))
        
        
        
    def plot_learning_curve(self, metric='loss'):
        
        fig, ax = plt.subplots()

        max_epoch = self.training_history[-1][1]['epoch'][-1]

        for i, (mode, history_dict) in enumerate(self.training_history):    
            ax.plot(history_dict['epoch'], history_dict[metric], color='blue', label=metric)
            ax.plot(history_dict['epoch'], history_dict[f'val_{metric}'], color='orange', label=f'val_{metric}')


            epochs = history_dict['epoch']
            mid_epoch =  (epochs[0] + epochs[-1])/2 - 0.5

            ax.text(mid_epoch/(max_epoch+1), 0.5, mode, transform=ax.transAxes)

            if i == 0:
                ax.set_xlim(0, max_epoch)

            else:
                ax.axvline(x=history_dict['epoch'][0], color='green')
                
                
        ax.set_xlabel('epoch')
        ax.set_ylabel(metric)
        
        return ax