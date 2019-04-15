#!/bin/bash
set -e -x

# Test transparent live migraiton of GPU App from GPU0 to GPU1
# Require: 
# - server: should have at least two GPUs, 
# - server: enable live-migraiton at server when start ./xserver.sh (default it's off)
#    MIGRATION=1 REPLY=1 ./xserver.sh  #best to set REPLY=1
# - client: start at least two App (tf-mnist_cnn, tf-mnist_mlp) almost concurrently
#    MIGRATION=1 REPLY=1 ./xclient.sh <App> #best to set REPLY=1
# Note, 
# - Live migration is transparent to App, thus have to parse logs to make sure it happens

# need to input DEV=<ip>:<dev-idx>, e.g. DEV=127.0.0.1:0
DEV=${DEV:-:0}
echo "GPU server is "$DEV

TEST1='tests/examples/tf/mnist_cnn.py'
TEST1_LOG='/tmp/migration_test_cnn.log'
TEST2='tests/examples/tf/mnist_mlp.py'
TEST2_LOG='/tmp/migration_test_mlp.log'
EVENT_MSG='Detect live migration event'

cd $(dirname $0)/..

if [ -f $TEST1_LOG ];then
  rm -f $TEST1_LOG
fi

if [ -f $TEST2_LOG ];then
  rm -f $TEST2_LOG
fi

echo -e '\n=========================='
echo ${BASH_SOURCE[0]}':Starting client1 of '$TEST1
DEV="${DEV}" MIGRATION=1 REPLY=1 ./xclient.sh python $TEST1 |tee -a $TEST1_LOG &
TEST1_ID=$!
echo ${BASH_SOURCE[0]}':PID1 is '$TEST1_ID
echo -e '==========================\n'

sleep 10
echo -e '==========================\n'
echo ${BASH_SOURCE[0]}':Starting client2 of '$TEST2
DEV="${DEV}" MIGRATION=1 REPLY=1 ./xclient.sh python $TEST2 |tee -a $TEST2_LOG  &
TEST2_ID=$!
echo ${BASH_SOURCE[0]}':PID2 is '$TEST2_ID
echo -e '==========================\n'

wait $TEST1_ID
  status1=$?

wait $TEST2_ID
  status2=$?

echo -e '\n\n==========================='
echo ${BASH_SOURCE[0]}': Client1 running status is '$status1
echo ${BASH_SOURCE[0]}': Client2 running status is '$status2

migFound=0
grep "$EVENT_MSG" $TEST1_LOG
if [ $? -eq 0 ];then
  echo ${BASH_SOURCE[0]}':====>Detect live migraiton in client1 log file '$TEST1_LOG
  migFound=1
fi 

grep "$EVENT_MSG" $TEST2_LOG > /dev/null
if [ $? -eq 0 ];then
  echo ${BASH_SOURCE[0]}':====>Detect live migraiton in client2 log file'$TEST2_LOG
  migFound=1
fi 

if [[ $status1 -eq 0 && $status2 -eq 0 && $migFound -eq 1 ]];then
  echo ${BASH_SOURCE[0]}": live migraiton is successful and test is passed"
  exit 0
else
  echo ${BASH_SOURCE[0]}": Faield to run Apps or live migraiton not detected"
  exit -1
fi
