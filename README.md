# Tensorflow Deep Learning 🤖📈📉

This is my exploration of [mrdbourke](https://github.com/mrdbourke)'s awesome 🔥 course on Deep Learning using Tensorflow.

## How did I approach learning?

> I wrote and rewrote all of the code myself. Yes, I am serious. Even though there will be much similarity with the original course notebooks, but I rewrote every markdown, every line of code in this entire repository. I took my time to:
>
>
>  ✔️ study everything deeply <br />
>  ✔️ develop my own routines with loads of useful functionalities! <br />
>  ✔️ made this repository into a well structured, and an installable Python package `src` <br />
>  ✔️ tracked issue, tasks, bugs and features like a `pro` 😎 using `Github Issues` and `Github Projects`

## Structure of this repository

```
.
|-- README.md                       # the README.md file of this repo 📖
|-- bugrefs                         # references for filing bugs 🐛
|-- checkpoints                     # model checkpointed weights for easy resuming of training 🏋️
|-- data                            # the datasets used in this repository 📈📉
|-- docs                            # docs for the deployed website of this repo 📗
|-- history_logs                    # logs of model training history useful for reloading models
|-- mkdocs.yml                      # Github Actions for making docs yml script 🌏
|-- models                          # TFSavedModel models trained on various deep learning tasks 🤖
|-- notebooks                       # the jupyter notebooks! 📚📚
|-- references                      # often images for easier insertion into jupyter notebooks
|-- reports                         # output reports of the analysis 🗒️
|-- requirements.txt                # the library requirements of the installable package (src) 📄
|-- scripts                         # mainly download scripts for data 📃
|-- setup.py                        # the setup script for installing the src package 
|-- src                             # the installable src package with useful routines
|-- tensorboard_logs                # tensorboard logs of model training for visualing on TensorBoard
```

## Structure of the `src` package

```
src/
|-- __init__.py                     # top level __init__.py
|-- evaluate                        # performance evaluation of models
|-- image                           # image processing routines
|-- models                          # custom models
|-- preprocess                      # general preprocessing routines
|-- text                            # text processing routines
|-- tfplay                          # TensorFlowPlayground reimplementation
|-- utils                           # general utilities
`-- visualize                       # visualization routines
```

## Fundamentals


|          	|   	|
|----------	|---	|
| **concepts** 	| `tensor algebra` `tensorflow-numpy link` `tensor manipulation` `constant tensor` `variable tensor` `random tensor` `sampling tensors` `tensor shapes intuition`                 `matrix multiplication intution` `aggregating tensors` `tensor datatypes` `tensor precision` `setting seed` `using GPUs with tensorflow`    	|
| **data**     	|   `constant tensors` `variable tensors` `drawing random tensors from probability distributions`	|
| **models**   	|   `None`	|

## Neural Network Regression

|            |            |
| ---------- | ---------- |
| concepts | `polynomial regression` `OLS` `sklearn LinearRegression` `basic tensorflow regression` `polynomial featurization` `creating tensorflow models` `improving tensorflow models` `evaluating tensorflow models` `loading tensorflow models` `saving tensorflow models` `box-cox transformation` |
| data | `polynomial model sampling` `medical cost ` |
| models | 🤖 **quadratic_regression:** `slr` `single_layer` `double_layer` `polyfeat` <br> 🤖 **medical_cost_prediction:** `3_layer_no_boxcox` `3_layer_boxcox` |

## Neural Network Classification

|            |            |
| ---------- | ---------- |
| **concepts** | `learning rate` `L1/L2 regularization` `activation functions` `gaussian noise` `sampling dummy data` `neurons` `layers` `learning curve` `decision boundary` `multiclass classification` `categorical crossentropy` `classification performance evaluation LearningRateScheduler` `feature engineering` |
| **data** | `fashion mnist` `circles` `exclusive_or` `gaussian` `spiral` |
| **models** | 🤖 **dummy_data_classification**: `TensorflowPlayground` <br> 🤖 **fashion_mnist**: `simple-dense-2layer` `medium-dense-2layer` `cnn` `cnn-best_lr` |

## Computer Vision

|            |            |
| ---------- | ---------- |
| **concepts** | `convolutional neural network` `parameter sharing` `data augmentation` `batch dataloader` `prefetching` `noise removal` `ClassicImageDataDirectory` `dropout regularization` `conv-pool conv-pool architecture` `LearningRateScheduler` `binary crossentropy` `categorical crossentropy` `classification report` `confusion matrix` |
| **data** | `pizza_steak` |
| **models** | 🤖 **pizza_steak_multiclass_classification**: `Dense` `TinyVGG` `TinyVGG-data-augment` `TinyVGG-data-augment-bestlr` `TinyVGG-data-augment-dropout-last` <br> 🤖 **10_food_multiclass_classification**: `TinyVGG` `TinyVGG-data-augment` `TinyVGG-data-augment-bestlr` `TinyVGG-Extra-Conv-Dense` `TinyVGG-Extra-Conv-BatchNorm-Dense` `TinyVGG-Extra-Conv-BatchNorm-Dense-ReduceLROnPlateau` `efficientnetb0_feature_extraction_1_percent` `efficientnetb0_feature_extraction_10_percent` `efficientnetb0_fine_tuning_10_percent` `efficientnetb0_fine_tuning_100_percent` |

## Transfer Learning

|            |            |
| ---------- | ---------- |
| **concepts** | `transfer learning` `ReduceLRonPlateau` `data augmentation as regularization` `as-is transfer learning` `feature-extraction transfer learning` `fine-tuning transfer learning` `TensorflowHub` `training callbacks` `TensorBoard` `ModelCheckpoint` `EarlyStopping` `ImageDataGenerator` `pretrained task vs downstream task` `gradual unfreezing` `keras Functional API` `GlobalAveragePool` `top-n accuracy` `GPU compute capability` `mixed precision training` `image normalization` `image resizing` `tf.data API` `map() ` `shuffle()` `batch()` `prefetch()` `sparse categorical crossentropy` |
| **data** | `10_food_classes` |
| **models** | 🤖 **10_food_multiclass_classification**: `efficientnetb0_feature_extraction_1_percent` `efficientnetb0_feature_extraction_10_percent` `efficientnetb0_fine_tuning_10_percent` `efficientnetb0_fine_tuning_100_percent` <br> 🤖 **101_food_multiclass_classification**: `efficientnetb0_fine_tune_10_percent (FoodVisionMini)` `resnet50v2_fine_tune_10_percent (FoodVisionMini)` `efficientnetb0_feature_extraction_all_data (FoodVisionBig)` ` efficientnetb0_fine_tune_all_data (FoodVisionBig)` |


## Natural Language Processing

|            |            |
| ---------- | ---------- |
| **concepts** | `Tf-idf` `count vectorization` `Multi-label classification` `machine translation` `seq2seq` `encoder-decoder` `attention` `ensembling` `text preprocessing` `text vectorization` `word-level tokenization` `character-level tokenization` `sub-word tokenization` `embeddings` `pretrained embeddings` `embedding layer` `embeddings as transfer learning` `high accuracy vs fast inference` `multimodal input models` `joint sentence classification` `Conv1D` `label smoothing` `TensorSliceDataset` `PrefetchDataset` `hybrid embeddings` `positional embeddings` `learning embedding representation of categorical feature` |
| **data** | `disaster_tweets` `PubMed_RCT` |
| **models** | 🤖 **disaster_tweets_classification**: `baseline-naive-bayes` `simple-dense` `GRU` `LSTM` `Bidirectional-LSTM` `CNN` `USE-Simple-Dense` `USE-Simple-Dense-10-percent` `ensemble-top3-avg` `ensemble-top3-majority` `ensemble-top3-meta-classifier` <br> 🤖 **pubmed_rct_abstract_multiclass_classification**: `naive-bayes-baseline` `USE-feature-extraction` `Conv1D-word-embed` `Conv1D-char-embed` `USE-char-hybrid-embed` `use-char-pos-embed-tribrid` |


## Time Series Forecasting

|            |            |
| ---------- | ---------- |
| **concepts** | `anomaly detection as classification` `forecasting as supervised regression` `train-test split for time series` `time series window-horizon` `multivariate time series` `data uncertainty` `model uncertainty` `open and closed systems` `turkey problem` `black swan events` `tensorflow Layer subclassing` `ensemble tricks –  variation on loss function` `window sizes` `seeds` `forecast uncertainty` |
| **data** | ``bitcoin_prices`` |
| **models** | 🤖 **bitcoin_time_series_prediction**: `naive-model-baseline_W1H1` `simple-dense_W7H1` `simple-dense_W30H1` `simple-dense_W30H7` `Conv1D_W7H1` `LSTM_W7H1` `multivariate-dense_W7H1` `NbeatsGeneric_W7H1` `ensemble-mean_W7H1` `ensemble-median_W7H1` `turkey-model_W7H1` |


## Original Resources


- [Original Repo](https://github.com/mrdbourke/tensorflow-deep-learning)
- [Youtube Playlist](https://www.youtube.com/playlist?list=PL6vjgQ2-qJFfU2vF6-lG9DlSa4tROkzt9)
- [Zero To Mastery - Original Course](https://academy.zerotomastery.io/p/learn-tensorflow)

