#!/bin/bash -e

cd $(dirname $0)

function run {
	echo "==== Running '$@' ===="
	$@
}

run python3 autoencoder/sae.py
run python3 mnist_nn/mnist.py
run python3 conv2d/conv2d.py
run python3 wordvec_subwords/wordvec_subwords.py
run python3 lstm_word/lstm_word.py
run python3 benchmark/train_mnist.py
run python3 benchmark/train_cifar10_resnet.py

echo -e "\n[*] All mxnet tests PASSED successfully!"

