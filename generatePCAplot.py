# -*- coding: utf-8 -*-
"""ECET380_Assgn2_Part2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pUleD1CS5SMVqYmTa_F7wKcZT-S9m2Ki
"""

# Commented out IPython magic to ensure Python compatibility.
# %load_ext autoreload
# %autoreload

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import preprocessDefinition
import importlib
!pip install -q -U keras-tuner
import kerastuner as kt
importlib.reload(preprocessDefinition)
from preprocessDefinition import preprocess
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.decomposition import PCA

dataset, info = tfds.load("oxford_flowers102", with_info=True, as_supervised=True)
dataset

info.features

class_names = info.features["label"].names
class_names

n_classes = info.features['label'].num_classes
n_classes

dataset_size = info.splits['train'].num_examples
dataset_size

train_set   = tfds.load('oxford_flowers102', split='train', as_supervised=True)
val_set     = tfds.load('oxford_flowers102', split='validation', as_supervised=True)
test_set     = tfds.load('oxford_flowers102', split='test', as_supervised=True)

batch_size = 32
test_set = test_set.map(preprocess).batch(batch_size).prefetch(1)

base_model = keras.applications.efficientnet.EfficientNetB0(
        weights = "imagenet",
        include_top = False)

# for layer in base_model.layers:
#     layer.trainable = False

avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(
    units = n_classes, 
    activation = "softmax")(avg)

model = keras.models.Model(
    inputs = base_model.input,
    outputs = output)

X2D = model.predict(test_set)

pca = PCA()
pca.fit(X2D)
var = pca.explained_variance_ratio_
var.cumsum()

plt.plot(var.cumsum())

plt.xlabel(" number of dimension")
plt.ylabel("explained variance")
plt.savefig("explainedVariancePlot.png")

plt.show()