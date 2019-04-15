#!/bin/bash -e

export MULTI_STREAM=0
export ASYNC_DtoH=0

cd /opt/caffe

./data/cifar10/get_cifar10.sh
./examples/cifar10/create_cifar10.sh
./examples/cifar10/train_quick.sh

./data/mnist/get_mnist.sh
./examples/mnist/create_mnist.sh
./examples/mnist/train_lenet.sh
./examples/mnist/train_mnist_autoencoder.sh

./examples/siamese/create_mnist_siamese.sh
./examples/siamese/train_mnist_siamese.sh


echo -e "\n[*] All Caffe tests PASSED successfully!"

