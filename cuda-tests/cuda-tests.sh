#!/bin/bash

set -e
source ./utils.sh

export MULTI_STREAM=0
export ASYNC_DtoH=0

# If no GPU address conflicts (limitation v5), e.g. 1) single-node multi-gpu; 2) multi-node multi-gpu + correct MASK_GB settings;
#   you can have PS-on-GPU tests:
#                  "./examples/binary/testcases/basic/p2pBandwidthLatencyTest.cu"
export ASAKA_TESTS_TO_SKIP="${ASAKA_TESTS_TO_SKIP} p2pBandwidthLatencyTest"

# CUDA test cases include both single-GPU and dual-GPU
# e.g. ./basic/deviceQueryDrv.cu can test on dual-GPU.
cd $(dirname $0)

run_cuda_test() {
  TARGET=/tmp/$(basename "${1}" .cu)
  rm -f "${TARGET}"

  PATH=$PATH:/usr/local/cuda/bin nvcc -ccbin g++ --std=c++11 -I$(dirname ${0})/examples/binary/cuda_common -m64 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_60,code=compute_60 -lcuda -lcudart -lcudnn -lcublas -lcufft -lcurand -o "$TARGET" "$@" $(find `dirname $1` -name "*.c") $(find `dirname $1` -name "*.cpp") || return false
  WS_BIN=$(dirname $1) "$TARGET"
}

start_tests() {
  evaluate_env_dev
  yellow "[TEST] Running CUDA tests with ${number_of_gpus} GPUs: ${DEV}"

  run_tests_in_folder ./examples/binary/testcases/ cu run_cuda_test
}
start_tests 2>&1

green "[TEST] All CUDA tests have passed!\n\n"
