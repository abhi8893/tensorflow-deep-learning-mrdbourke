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
|-- __init__.py                     # top level \_\_init\_\_.py
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



## Original Resources


- [Original Repo](https://github.com/mrdbourke/tensorflow-deep-learning)
- [Youtube Playlist](https://www.youtube.com/playlist?list=PL6vjgQ2-qJFfU2vF6-lG9DlSa4tROkzt9)
- [Zero To Mastery - Original Course](https://academy.zerotomastery.io/p/learn-tensorflow)

