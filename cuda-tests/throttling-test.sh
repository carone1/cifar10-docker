#!/bin/bash
# Test GPU throttling: run 2 Tensorflow-AlexNet clients with different GPU quota settings, both shall run to complete but get different performance
# Steps: 
# - server: start Asaka server with throttling enabled, i.e., set VGPU_QUOTA <1, here assume server is already started
#    RDMAV_FORK_SAFE=1 VGPU_QUOTA=0.50 DEV=:0 ./xserver.sh
# - client: start two clients, running same model with different quota settings, i.e., 25% and 75%

set -e -x
DEV=${DEV:-:0}
echo "GPU server is "$DEV
cd $(dirname $0)/..
QUOTA1=25
QUOTA2=75
KEYWORDS='total images/sec'
LOG1='/tmp/throttle-test-alexnet1.log'
LOG2='/tmp/throttle-test-alexnet2.log'
Diff1=1.5
testF='tests/examples/tf/benchmarks/tf_cnn_benchmarks/tf_cnn_benchmarks.py'

#client1, run TensorFlow-AlexNet, with 25% GPU resource
GMEM=4 DEV=$DEV/$QUOTA1 RDMAV_FORK_SAFE=1 python3 $testF --model alexnet --batch_size 64 --num_batches 200 2>&1 |tee -a $LOG1
#client2, start client2, same model as above, with 75% GPU resource
GMEM=4 DEV=$DEV/$QUOTA2 RDMAV_FORK_SAFE=1 python3 $testF --model alexnet --batch_size 64 --num_batches 500 2>&1 |tee -a $LOG2

sleep 10

# both shall run to complete, get training perf as integer 
perf1=`grep 'total images/sec' $LOG1 |awk -F ':' '{print $2}'|awk '{print int($0)}'`
perf2=`grep 'total images/sec' $LOG2 |awk -F ':' '{print $2}'|awk '{print int($0)}'`
rm -f $LOG1 $LOG2

# compare the perf 
rst1=`awk 'BEGIN {print '$perf1' * '$Diff1';}' | awk '{print int($0)}'`
if [ $perf2 -gt $rst1 ];then
  echo "Pass throttling test, client1:$perf1 client2:$perf2, perf diff meets pre-defined criterion: $Diff1 ($rst1)"
  exit 0
else
  echo "Failed in throttling test, client1:$perf1 client2:$perf2, such perf do not meet pre-defined criterion: $Diff1 diff ($rst1)"
  exit -1
fi
