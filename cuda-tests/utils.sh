#!/bin/bash

GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
END="\033[0m"

# Parameters
#    1: color
#    2: string need to be colored
colorize() {
  echo -e "${!1}${2}${END}"
}

red() {
  colorize RED "${1}"
  exit 1
}

green() {
  colorize GREEN "${1}"
}

yellow() {
  colorize YELLOW "${1}"
}


# Get GPU address array and number of GPUs
evaluate_env_dev() {
  IFS=',' read -ra array_of_gpus <<< "${DEV}"
  number_of_gpus=${#array_of_gpus[@]}
}

# Start particular test case and print out result
# Parameters:
#   1: function being used to trigger this test
#   2: path of file being tested
run_test() {
  func=${1}
  file=${2}

  yellow "[TEST] Running test:"
  yellow $@

  filename=$(basename ${file%.*} || basename ${file})
  if [[ -z ${ASAKA_TESTS_TO_RUN} && ! ${ASAKA_TESTS_TO_SKIP} = *${filename}* || \
        ${ASAKA_TESTS_TO_RUN} = *${filename}* ]]; then
    if $@ ; then
      green "${file} passed!\n"
    else
      red "${file} failed\n"
    fi
  fi
}

# Parameters:
#    1: folder's abosolute path
#    2: test file extension
#    3: function being used to run test
run_tests_in_folder() {
  folder=${1}
  extension=${2}
  func=${3}

  test_files=$(find ${folder} -name *.${extension} -type f)

  for file in ${test_files}; do
    run_test ${func} ${file}
  done
}

# Parameters:
#    1: array that contains tests
#    2: function being used to run test
run_tests_in_array() {
  tests=${1}

  for t in "${tests[@]}"; do
    run_test "" ${t}
    echo "sleeping for 5 seconds to fully release vGPUs"
    sleep 5
  done
}
