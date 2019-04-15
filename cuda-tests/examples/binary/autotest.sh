#!/bin/bash

export MULTI_STREAM=0
export ASYNC_DtoH=0

do_test() {
  echo -e "\n[*] Testing CUDA application - $@"
  TARGET=/tmp/$(basename "$1" .cu)
  rm -f "${TARGET}"
  
  PATH=$PATH:/usr/local/cuda/bin nvcc -ccbin g++ --std=c++11 -I$(dirname $0)/cuda_common -m64 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_60,code=compute_60 -lcuda -lcudart -lcudnn -lcublas -lcufft -lcurand -o "$TARGET" "$@" $(find `dirname $1` -name "*.c") $(find `dirname $1` -name "*.cpp") || return false
  WS_BIN=$(dirname $1) "$TARGET"
}

check_err() {
  ! do_test "$@" && echo -e "\n[x] Test case '$@' failed." && exit 1
}

if [[ "$@" != "" ]]; then
  check_err "$@"
else
  for F in $(dirname $0)/testcases/[a-zA-Z]*/*.cu; do
    check_err "$F"
  done 2>&1
fi

echo -e "\n[*] All binary tests PASSED successfully!"

