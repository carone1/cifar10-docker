#!/bin/bash -e

cd $(dirname $0)

function run {
	echo "==== Running '$@' ===="
	$@
}

run ./basic/mnist_cnn.py
run ./basic/tf_eager.py
run ./basic/tf_rnn.py
run ./nn_models/alexnet_benchmark.py
run ./nn_models/autoencoder_runner.py
run ./nn_models/cifar10_multi_gpu_train.py
run ./benchmarks/run inception3 cpu
run ./benchmarks/run resnet50 cpu
run ./benchmarks/run lenet cpu
run ./benchmarks/run resnet50 gpu
run ./benchmarks/run googlenet gpu

echo -e "\n[*] All tensorflow tests PASSED successfully!"

