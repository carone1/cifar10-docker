#!/bin/bash -e

cd $(dirname $0)

function run {
	echo "==== Running '$@' ===="
	$@
}

run ./offical/mnist_mlp.py
run ./offical/mnist_cnn.py
run ./offical/reuters_mlp.py
run ./offical/cifar10_cnn.py
run ./inference/vgg16.py
run ./multi_gpu_model/resnet20v2.py

echo -e "\n[*] All keras tests PASSED successfully!"

