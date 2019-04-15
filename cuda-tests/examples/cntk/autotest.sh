#!/bin/bash -e

cd $(dirname $0)

function run {
	echo "==== Running '$@' ===="
	$@
}

./__download_dataset__.sh

run ./LogisticRegression_FunctionalAPI.py
run ./LogisticRegression_GraphAPI.py
run ./SimpleMNIST.py
run ./ConvNetLRN_CIFAR10_DataAug.py
run ./InceptionV3_ImageNet.py

echo -e "\n[*] All CNTK tests PASSED successfully!"

