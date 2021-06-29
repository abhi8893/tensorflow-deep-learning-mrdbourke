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

## Original Resources


- [Original Repo](https://github.com/mrdbourke/tensorflow-deep-learning)
- [Youtube Playlist](https://www.youtube.com/playlist?list=PL6vjgQ2-qJFfU2vF6-lG9DlSa4tROkzt9)
- [Zero To Mastery - Original Course](https://academy.zerotomastery.io/p/learn-tensorflow)

