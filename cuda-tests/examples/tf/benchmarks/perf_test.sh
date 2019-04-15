#!/bin/bash -e

export GMEM=15

for MODEL in resnet50 inception3 alexnet resnet152 inception4 vgg16 lenet googlenet; do
  printf "%14s:" $MODEL
  printf " cpu = %s" $(`dirname $0`/run ${MODEL} cpu | grep "total images/sec" | awk '{print $NF}')
  printf " gpu = %s" $(`dirname $0`/run ${MODEL} gpu | grep "total images/sec" | awk '{print $NF}')
  echo
done 2>/tmp/logs.stderr

