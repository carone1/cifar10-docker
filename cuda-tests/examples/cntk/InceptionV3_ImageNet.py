#!/usr/bin/env python3
import os
os.chdir('/tmp/cntk')

# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from cntk.initializer import he_normal
from cntk.layers import AveragePooling, MaxPooling, BatchNormalization, Convolution, Dense, Dropout
from cntk.ops import minus, element_times, relu, splice

#
# This file contains the basic build block of Inception Network as defined in:
#
#   https://arxiv.org/pdf/1512.00567.pdf
#
# and in Tensorflow implementation
#

#
# Convolution layer with Batch Normalization and Rectifier Linear activation.
#
def conv_bn_relu_layer(input, num_filters, filter_size, strides=(1,1), pad=True, bnTimeConst=4096, init=he_normal()):
    conv = Convolution(filter_size, num_filters, activation=None, init=init, pad=pad, strides=strides, bias=False)(input)
    bn   = BatchNormalization(map_rank=1, normalization_time_constant=bnTimeConst, use_cntk_engine=False)(conv)
    return relu(bn)

#
# Figure 5 from https://arxiv.org/pdf/1512.00567.pdf
# Modified with the added 5x5 branch to match Tensorflow implementation
#
def inception_block_1(input, num1x1, num5x5, num3x3dbl, numPool, bnTimeConst):

    # 1x1 Convolution
    branch1x1 = conv_bn_relu_layer(input, num1x1, (1,1), (1,1), True, bnTimeConst)

    # 5x5 Convolution
    branch5x5_1 = conv_bn_relu_layer(input, num5x5[0], (1,1), (1,1), True, bnTimeConst)
    branch5x5   = conv_bn_relu_layer(branch5x5_1, num5x5[1], (5,5), (1,1), True, bnTimeConst)

    # Double 3x3 Convolution
    branch3x3dbl_1 = conv_bn_relu_layer(input, num3x3dbl[0], (1,1), (1,1), True, bnTimeConst)
    branch3x3dbl_2 = conv_bn_relu_layer(branch3x3dbl_1, num3x3dbl[1], (3,3), (1,1), True, bnTimeConst)
    branch3x3dbl   = conv_bn_relu_layer(branch3x3dbl_2, num3x3dbl[2], (3,3), (1,1), True, bnTimeConst)

    # Average Pooling
    branchPool_avgpool = AveragePooling((3,3), strides=(1,1), pad=True)(input)
    branchPool = conv_bn_relu_layer(branchPool_avgpool, numPool, (1,1), (1,1), True, bnTimeConst)

    out = splice(branch1x1, branch5x5, branch3x3dbl, branchPool, axis=0)

    return out

def inception_block_2(input, num3x3, num3x3dbl, bnTimeConst):

    # 3x3 Convolution
    branch3x3 = conv_bn_relu_layer(input, num3x3, (3,3), (2,2), False, bnTimeConst)

    # Double 3x3 Convolution
    branch3x3dbl_1 = conv_bn_relu_layer(input, num3x3dbl[0], (1,1), (1,1), True, bnTimeConst)
    branch3x3dbl_2 = conv_bn_relu_layer(branch3x3dbl_1, num3x3dbl[1], (3,3), (1,1), True, bnTimeConst)
    branch3x3dbl   = conv_bn_relu_layer(branch3x3dbl_2, num3x3dbl[2], (3,3), (2,2), False, bnTimeConst)

    # Max Pooling
    branchPool = MaxPooling((3,3), strides=(2,2), pad=False)(input)

    out = splice(branch3x3, branch3x3dbl, branchPool, axis=0)

    return out

#
# Figure 6 from https://arxiv.org/pdf/1512.00567.pdf
#
def inception_block_3(input, num1x1, num7x7, num7x7dbl, numPool, bnTimeConst):

    # 1x1 Convolution
    branch1x1 = conv_bn_relu_layer(input, num1x1, (1,1), (1,1), True, bnTimeConst)

    # 7x7 Convolution
    branch7x7_1 = conv_bn_relu_layer(input, num7x7[0], (1,1), (1,1), True, bnTimeConst)
    branch7x7_2 = conv_bn_relu_layer(branch7x7_1, num7x7[1], (1,7), (1,1), True, bnTimeConst)
    branch7x7   = conv_bn_relu_layer(branch7x7_2, num7x7[2], (7,1), (1,1), True, bnTimeConst)

    # Double 7x7 Convolution
    branch7x7dbl_1 = conv_bn_relu_layer(input, num7x7dbl[0], (1,1), (1,1), True, bnTimeConst)
    branch7x7dbl_2 = conv_bn_relu_layer(branch7x7dbl_1, num7x7dbl[1], (7,1), (1,1), True, bnTimeConst)
    branch7x7dbl_3 = conv_bn_relu_layer(branch7x7dbl_2, num7x7dbl[2], (1,7), (1,1), True, bnTimeConst)
    branch7x7dbl_4 = conv_bn_relu_layer(branch7x7dbl_3, num7x7dbl[3], (7,1), (1,1), True, bnTimeConst)
    branch7x7dbl   = conv_bn_relu_layer(branch7x7dbl_4, num7x7dbl[4], (1,7), (1,1), True, bnTimeConst)

    # Average Pooling
    branchPool_avgpool = AveragePooling((3,3), strides=(1,1), pad=True)(input)
    branchPool = conv_bn_relu_layer(branchPool_avgpool, numPool, (1,1), (1,1), True, bnTimeConst)

    out = splice(branch1x1, branch7x7, branch7x7dbl, branchPool, axis=0)

    return out

def inception_block_4(input, num3x3, num7x7_3x3, bnTimeConst):

    # 3x3 Convolution
    branch3x3_1 = conv_bn_relu_layer(input, num3x3[0], (1,1), (1,1), True, bnTimeConst)
    branch3x3   = conv_bn_relu_layer(branch3x3_1, num3x3[1], (3,3), (2,2), False, bnTimeConst)

    # 7x7 3x3 Convolution
    branch7x7_3x3_1 = conv_bn_relu_layer(input, num7x7_3x3[0], (1,1), (1,1), True, bnTimeConst)
    branch7x7_3x3_2 = conv_bn_relu_layer(branch7x7_3x3_1, num7x7_3x3[1], (1,7), (1,1), True, bnTimeConst)
    branch7x7_3x3_3 = conv_bn_relu_layer(branch7x7_3x3_2, num7x7_3x3[2], (7,1), (1,1), True, bnTimeConst)
    branch7x7_3x3   = conv_bn_relu_layer(branch7x7_3x3_3, num7x7_3x3[3], (3,3), (2,2), False, bnTimeConst)

    # Max Pooling
    branchPool = MaxPooling((3,3), strides=(2,2), pad=False)(input)

    out = splice(branch3x3, branch7x7_3x3, branchPool, axis=0)

    return out

#
# Figure 7 from https://arxiv.org/pdf/1512.00567.pdf
#
def inception_block_5(input, num1x1, num3x3, num3x3_3x3, numPool, bnTimeConst):

    # 1x1 Convolution
    branch1x1 = conv_bn_relu_layer(input, num1x1, (1,1), (1,1), True, bnTimeConst)

    # 3x3 Convolution
    branch3x3_1 = conv_bn_relu_layer(input, num3x3[0], (1,1), (1,1), True, bnTimeConst)
    branch3x3_2 = conv_bn_relu_layer(branch3x3_1, num3x3[1], (1,3), (1,1), True, bnTimeConst)
    branch3x3_3 = conv_bn_relu_layer(branch3x3_1, num3x3[2], (3,1), (1,1), True, bnTimeConst)
    branch3x3   = splice(branch3x3_2, branch3x3_3, axis=0)

    # 3x3 3x3 Convolution
    branch3x3_3x3_1 = conv_bn_relu_layer(input, num3x3_3x3[0], (1,1), (1,1), True, bnTimeConst)
    branch3x3_3x3_2 = conv_bn_relu_layer(branch3x3_3x3_1, num3x3_3x3[1], (3,3), (1,1), True, bnTimeConst)
    branch3x3_3x3_3 = conv_bn_relu_layer(branch3x3_3x3_2, num3x3_3x3[1], (1,3), (1,1), True, bnTimeConst)
    branch3x3_3x3_4 = conv_bn_relu_layer(branch3x3_3x3_2, num3x3_3x3[3], (3,1), (1,1), True, bnTimeConst)
    branch3x3_3x3   = splice(branch3x3_3x3_3, branch3x3_3x3_4, axis=0)

    # Average Pooling
    branchPool_avgpool = AveragePooling((3,3), strides=(1,1), pad=True)(input)
    branchPool = conv_bn_relu_layer(branchPool_avgpool, numPool, (1,1), (1,1), True, bnTimeConst)

    out = splice(branch1x1, branch3x3, branch3x3_3x3, branchPool, axis=0)

    return out


#
# Inception V3 model with normalized input, to use the below function
# remove "ImageNet1K_mean.xml" from each reader.
#
def inception_v3_norm_model(input, labelDim, dropRate, bnTimeConst):

    # Normalize inputs to -1 and 1.
    featMean  = 128
    featScale = 1/128
    input_subtracted = minus(input, featMean)
    input_scaled = element_times(input_subtracted, featScale)

    return inception_v3_model(input_scaled, labelDim, dropRate, bnTimeConst)

#
# Inception V3 model
#
def inception_v3_model(input, labelDim, dropRate, bnTimeConst):

    # 299 x 299 x 3
    conv1 = conv_bn_relu_layer(input, 32, (3,3), (2,2), False, bnTimeConst)
    # 149 x 149 x 32
    conv2 = conv_bn_relu_layer(conv1, 32, (3,3), (1,1), False, bnTimeConst)
    # 147 x 147 x 32
    conv3 = conv_bn_relu_layer(conv2, 64, (3,3), (1,1), True, bnTimeConst)
    # 147 x 147 x 64
    pool1 = MaxPooling(filter_shape=(3,3), strides=(2,2), pad=False)(conv3)
    # 73 x 73 x 64
    conv4 = conv_bn_relu_layer(pool1, 80, (1,1), (1,1), False, bnTimeConst)
    # 73 x 73 x 80
    conv5 = conv_bn_relu_layer(conv4, 192, (3,3), (1,1), False, bnTimeConst)
    # 71 x 71 x 192
    pool2 = MaxPooling(filter_shape=(3,3), strides=(2,2), pad=False)(conv5)
    # 35 x 35 x 192

    #
    # Inception Blocks
    #
    mixed1 = inception_block_1(pool2, 64, [48, 64], [64, 96, 96], 32, bnTimeConst)
    # 35 x 35 x 256
    mixed2 = inception_block_1(mixed1, 64, [48, 64], [64, 96, 96], 64, bnTimeConst)
    # 35 x 35 x 288
    mixed3 = inception_block_1(mixed2, 64, [48, 64], [64, 96, 96], 64, bnTimeConst)
    # 35 x 35 x 288
    mixed4 = inception_block_2(mixed3, 384, [64, 96, 96], bnTimeConst)
    # 17 x 17 x 768
    mixed5 = inception_block_3(mixed4, 192, [128, 128, 192], [128, 128, 128, 128, 192], 192, bnTimeConst)
    # 17 x 17 x 768
    mixed6 = inception_block_3(mixed5, 192, [160, 160, 192], [160, 160, 160, 160, 192], 192, bnTimeConst)
    # 17 x 17 x 768
    mixed7 = inception_block_3(mixed6, 192, [160, 160, 192], [160, 160, 160, 160, 192], 192, bnTimeConst)
    # 17 x 17 x 768
    mixed8 = inception_block_3(mixed7, 192, [192, 192, 192], [192, 192, 192, 192, 192], 192, bnTimeConst)
    # 17 x 17 x 768
    mixed9 = inception_block_4(mixed8, [192, 320], [192, 192, 192, 192], bnTimeConst)
    # 8 x 8 x 1280
    mixed10 = inception_block_5(mixed9, 320, [384, 384, 384], [448, 384, 384, 384], 192, bnTimeConst)
    # 8 x 8 x 2048
    mixed11 = inception_block_5(mixed10, 320, [384, 384, 384], [448, 384, 384, 384], 192, bnTimeConst)
    # 8 x 8 x 2048

    #
    # Prediction
    #
    pool3 = AveragePooling(filter_shape=(8,8), pad=False)(mixed11)
    # 1 x 1 x 2048
    drop = Dropout(dropout_rate=dropRate)(pool3)
    # 1 x 1 x 2048
    z = Dense(labelDim, init=he_normal())(drop)

    #
    # Auxiliary
    #
    # 17 x 17 x 768
    auxPool =  AveragePooling(filter_shape=(5,5), strides=(3,3), pad=False)(mixed8)
    # 5 x 5 x 768
    auxConv1 = conv_bn_relu_layer(auxPool, 128, (1,1), (1,1), True, bnTimeConst)
    # 5 x 5 x 128
    auxConv2 = conv_bn_relu_layer(auxConv1, 768, (5,5), (1,1), False, bnTimeConst)
    # 1 x 1 x 768
    aux = Dense(labelDim, init=he_normal())(auxConv2)

    return {
        'z':   z,
        'aux': aux
    }

# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import math
import argparse
import numpy as np
import cntk as C

# default Paths relative to current python file.
abs_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(abs_path, "dataset") # os.path.join(abs_path, "..", "..", "..", "..", "DataSets", "ImageNet")
config_path = abs_path
os.chdir(data_path)

# model dimensions
IMAGE_HEIGHT = 299
IMAGE_WIDTH = 299
NUM_CHANNELS = 3 # RGB
NUM_CLASSES = 1000
model_name = "InceptionV3.model"

# Create a minibatch source.
def create_image_mb_source(map_file, is_training, total_number_of_samples):
    if not os.path.exists(map_file):
        raise RuntimeError("File '%s' does not exist." % (map_file))

    # transformation pipeline for the features has jitter/crop only when training
    transforms = []
    if is_training:
        transforms += [
            C.io.transforms.crop(crop_type='randomarea', area_ratio=(0.05, 1.0), aspect_ratio=(0.75, 1.0), jitter_type='uniratio'), # train uses jitter
            C.io.transforms.scale(width=IMAGE_WIDTH, height=IMAGE_HEIGHT, channels=NUM_CHANNELS, interpolations='linear'),
            C.io.transforms.color(brightness_radius=0.125, contrast_radius=0.5, saturation_radius=0.5)
        ]
    else:
        transforms += [
            C.io.transforms.crop(crop_type='center', side_ratio=0.875), # test has no jitter
            C.io.transforms.scale(width=IMAGE_WIDTH, height=IMAGE_HEIGHT, channels=NUM_CHANNELS, interpolations='linear')
        ]

    # deserializer
    return C.io.MinibatchSource(
        C.io.ImageDeserializer(map_file, C.io.StreamDefs(
            features=C.io.StreamDef(field='image', transforms=transforms), # first column in map file is referred to as 'image'
            labels=C.io.StreamDef(field='label', shape=NUM_CLASSES))),   # and second as 'label'
        randomize=is_training,
        max_samples=total_number_of_samples,
        multithreaded_deserializer=True)

# Create the network.
def create_inception_v3():

    # Input variables denoting the features and label data
    feature_var = C.ops.input_variable((NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH))
    label_var = C.ops.input_variable((NUM_CLASSES))

    drop_rate = 0.2
    bn_time_const = 4096
    out = inception_v3_norm_model(feature_var, NUM_CLASSES, drop_rate, bn_time_const)

    # loss and metric
    aux_weight = 0.3
    ce_aux = C.losses.cross_entropy_with_softmax(out['aux'], label_var)
    ce_z = C.losses.cross_entropy_with_softmax(out['z'], label_var)
    ce = C.ops.plus(C.ops.element_times(ce_aux, aux_weight), ce_z)
    pe = C.metrics.classification_error(out['z'], label_var)
    pe5 = C.metrics.classification_error(out['z'], label_var, topN=5)

    C.logging.log_number_of_parameters(out['z'])
    print()

    return {
        'feature'   : feature_var,
        'label'     : label_var,
        'ce'        : ce,
        'pe'        : pe,
        'pe5'       : pe5,
        'output'    : out['z'],
        'outputAux' : out['aux']
    }

# Create trainer
def create_trainer(network, epoch_size, num_epochs, minibatch_size):

    # CNTK weights new gradient by (1-momentum) for unit gain,
    # thus we divide Caffe's learning rate by (1-momentum)
    initial_learning_rate = 0.45 # equal to 0.045 in caffe
    initial_learning_rate *= minibatch_size / 32
    learn_rate_adjust_interval = 2
    learn_rate_decrease_factor = 0.94

    # Set learning parameters
    lr_per_mb = []
    learning_rate = initial_learning_rate
    for i in range(0, num_epochs, learn_rate_adjust_interval):
        lr_per_mb.extend([learning_rate] * learn_rate_adjust_interval)
        learning_rate *= learn_rate_decrease_factor

    lr_schedule = C.learners.learning_parameter_schedule(lr_per_mb, epoch_size=epoch_size)
    mm_schedule = C.learners.momentum_schedule(0.9)
    l2_reg_weight = 0.0001 # CNTK L2 regularization is per sample, thus same as Caffe

    # Create learner
    learner = C.learners.nesterov(network['ce'].parameters, lr_schedule, mm_schedule,
                                  l2_regularization_weight=l2_reg_weight)

    # Create trainer
    return C.train.Trainer(network['output'], (network['ce'], network['pe']), learner)

# Train and test
def train_and_test(network, trainer, train_source, test_source, progress_printer, max_epochs, minibatch_size, epoch_size, restore, profiler_dir, testing_parameters):

    # define mapping from intput streams to network inputs
    input_map = {
        network['feature']: train_source.streams.features,
        network['label']: train_source.streams.labels
    }


    for epoch in range(max_epochs):       # loop over epochs
        sample_count = 0
        while sample_count < epoch_size:  # loop over minibatches in the epoch
            data = train_source.next_minibatch(min(minibatch_size, epoch_size-sample_count), input_map=input_map) # fetch minibatch.
            trainer.train_minibatch(data)                                   # update model with it
            sample_count += trainer.previous_minibatch_sample_count         # count samples processed so far
            progress_printer.update_with_trainer(trainer, with_metric=True) # log progress
        progress_printer.epoch_summary(with_metric=True)

    # Finished
    # Evaluation parameters
    test_epoch_size, test_minibatch_size = testing_parameters

    # process minibatches and evaluate the model
    metric_numer = 0
    metric_denom = 0
    sample_count = 0
    minibatch_index = 0

    while sample_count < test_epoch_size:
        current_minibatch = min(test_minibatch_size, test_epoch_size - sample_count)
        # Fetch next test min batch.
        data = test_source.next_minibatch(current_minibatch, input_map=input_map)
        # minibatch data to be trained with
        metric_numer += trainer.test_minibatch(data) * current_minibatch
        metric_denom += current_minibatch
        # Keep track of the number of samples processed so far.
        sample_count += data[network['label']].num_samples
        minibatch_index += 1

    print("")
    print("Final Results: Minibatch[1-{}]: errs = {:0.2f}% * {}".format(minibatch_index+1, (metric_numer*100.0)/metric_denom, metric_denom))
    print("")

    return metric_numer/metric_denom

# Train and evaluate the network.
def inception_v3_train_and_eval(train_data, test_data, minibatch_size=32, epoch_size=1281167, max_epochs=300, 
                                restore=True, log_to_file=None, num_mbs_per_log=100, gen_heartbeat=False, profiler_dir=None, testing_parameters=(5000, 32)):
    C.debugging.set_computation_network_trace_level(1)

    progress_printer = C.logging.ProgressPrinter(
        freq=num_mbs_per_log,
        tag='Training',
        log_to_file=log_to_file,
        gen_heartbeat=gen_heartbeat,
        num_epochs=max_epochs)

    network = create_inception_v3()
    trainer = create_trainer(network, epoch_size, max_epochs, minibatch_size)
    train_source = create_image_mb_source(train_data, True, total_number_of_samples=max_epochs * epoch_size)
    test_source = create_image_mb_source(test_data, False, total_number_of_samples=C.io.FULL_DATA_SWEEP)
    return train_and_test(network, trainer, train_source, test_source, progress_printer, max_epochs, minibatch_size, epoch_size, restore, profiler_dir, testing_parameters)


if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-datadir', '--datadir', help='Data directory where the cifar-10 dataset is located', required=False, default=data_path)
    parser.add_argument('-configdir', '--configdir', help='Config directory where this python script is located', required=False, default=config_path)
    parser.add_argument('-outputdir', '--outputdir', help='Output directory for checkpoints and models', required=False, default=None)
    parser.add_argument('-profilerdir', '--profilerdir', help='Directory for saving profiler output', required=False, default=None)
    parser.add_argument('-logdir', '--logdir', help='Log file', required=False, default=None)
    parser.add_argument('-n', '--num_epochs', help='Total number of epochs to train', type=int, required=False, default='2')
    parser.add_argument('-m', '--minibatch_size', help='Minibatch size', type=int, required=False, default='32')
    parser.add_argument('-e', '--epoch_size', help='Epoch size', type=int, required=False, default='5000')
    parser.add_argument('-r', '--restart', help='Indicating whether to restart from scratch (instead of restart from checkpoint file by default)', action='store_true')
    parser.add_argument('-device', '--device', type=int, help="Force to run the script on a specified device", required=False, default=None)

    args = vars(parser.parse_args())

    if args['device'] is not None:
        C.device.try_set_default_device(C.device.gpu(args['device']))

    data_path = args['datadir']

    if not os.path.isdir(data_path):
        raise RuntimeError("Directory %s does not exist" % data_path)

    os.chdir(data_path)

    train_data = os.path.join(data_path, 'train_map.txt')
    test_data = os.path.join(data_path, 'train_map.txt')

    inception_v3_train_and_eval(train_data, test_data,
                                minibatch_size=args['minibatch_size'],
                                epoch_size=args['epoch_size'],
                                max_epochs=args['num_epochs'],
                                restore=not args['restart'],
                                log_to_file=args['logdir'],
                                num_mbs_per_log=100,
                                gen_heartbeat=True,
                                profiler_dir=args['profilerdir'])
