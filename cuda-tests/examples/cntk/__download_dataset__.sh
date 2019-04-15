#!/bin/sh

mkdir -p /tmp/cntk
exec curl -L https://github.com/ghostplant/public/releases/download/firewall/cntk-dataset.tgz | tar xzvf - -C /tmp/cntk >/dev/null
