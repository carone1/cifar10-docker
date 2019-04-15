#!/bin/bash -e

cd $(dirname $0)

for I in ./*/autotest.sh; do
	echo -e "\n==== Testing '$(dirname $I)' ===="
	$I || exit 1
done

echo -e "\n[*] All test cases PASSED successfully!"

