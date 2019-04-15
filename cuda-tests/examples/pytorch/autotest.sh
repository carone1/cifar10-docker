#!/bin/bash -e

cd $(dirname $0)

function run {
	echo "==== Running '$@' ===="
	$@
}

run python3 mnist/main.py
run python3 examples_autograd/dynamic_net.py
run python3 vae/main.py
run python3 regression/main.py
run python3 cifar10/cifar10.py

echo -e "\n[*] All pytorch tests PASSED successfully!"

