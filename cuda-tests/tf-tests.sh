#!/bin/bash

set -e
source ./utils.sh

cd $(dirname $0)

start_tests() {
  evaluate_env_dev
  yellow "[TEST] Running TensorFlow tests with ${number_of_gpus} GPUs"

  declare -a tests=("./examples/tf/basic/mnist_cnn.py"
                    "./examples/tf/basic/tf_eager.py"
                    "./examples/tf/basic/tf_rnn.py"
                    "./examples/tf/nn_models/alexnet_benchmark.py"
                    "./examples/tf/nn_models/autoencoder_runner.py"
                    "./examples/tf/nn_models/cifar10_multi_gpu_train.py"
                    "./examples/tf/benchmarks/run inception3 cpu"
                    "./examples/tf/benchmarks/run resnet50 cpu"
                    "./examples/tf/benchmarks/run lenet cpu"
                    # The following 2 requires Tensorflow built with Asaka patch
                    "./examples/tf/benchmarks/run resnet50 gpu"
                    "./examples/tf/benchmarks/run googlenet gpu")

  # In the future: mpiexec --allow-run-as-root -np 4 python3 ./examples/tf/benchmarks/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model=resnet50 --batch_size=64 --variable_update=horovod --horovod_device=gpu --num_batches=2000

  run_tests_in_array ${tests}
}
start_tests 2>&1

green "[TEST] All TensorFlow tests have passed!\n\n"
