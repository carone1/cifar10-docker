#!/usr/bin/python3

import os
os.environ['GMEM'] = '12'

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import keras
from keras.applications import ResNet50, InceptionV3
from keras.utils import multi_gpu_model
import numpy as np
from keras import backend as K

height = 224
width = 224

num_samples = 7964
val_samples = 1738
num_classes = 5

# Instantiate the base model (or "template" model).
# We recommend doing this with under a CPU device scope,
# so that the model's weights are hosted on CPU memory.
# Otherwise they may end up hosted on a GPU, which would
# complicate weight sharing.

if K.image_data_format() == 'channels_first':
    image_shape = (3, height, width)
else:
    image_shape = (height, width, 3)

with tf.device('/cpu:0'):
    model = InceptionV3(weights=None,
                    include_top=True,
                    input_shape=image_shape,
                    classes=num_classes)

model = multi_gpu_model(model)


model.compile(loss='categorical_crossentropy',
		               optimizer='rmsprop')

model.summary()


from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator()
# gen = datagen.flow_from_directory('/root/PetImages', target_size=(height, width), batch_size=128)

x = np.random.random((num_samples, ) + image_shape)
y = np.random.random((num_samples, num_classes))
gen = datagen.flow(x, y, batch_size=128)

# This `fit` call will be distributed on 8 GPUs.
# Since the batch size is 256, each GPU will process 32 samples.
# os.environ['WIDTH'] = '0'

model.fit_generator(gen, epochs=2)

# val_gen = datagen.flow_from_directory('/root/PetValidation', target_size=(height, width), batch_size=128)
# outs = model.predict_generator(val_gen)
# print(outs)
