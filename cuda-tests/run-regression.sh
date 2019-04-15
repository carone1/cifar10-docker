#!/bin/bash
set -e

test_dir=`dirname $0`
export CUDADIR=$test_dir/..
export ARCH=`basename ${CUDADIR}`
WS=.

pushd $test_dir
  ulimit -c unlimited

  if [ -z ${ASAKA_TESTS_CUDA} ]; then
    timeout 18000 ./cuda-tests.sh
  fi

  if [ -z ${ASAKA_TESTS_TF} ]; then
   timeout 18000 ./tf-tests.sh

   timeout 18000 ./examples/pytorch/autotest.sh
   timeout 18000 ./examples/caffe/autotest.sh
   timeout 18000 ./examples/mxnet/autotest.sh
   timeout 18000 ./examples/cntk/autotest.sh
  fi

popd
