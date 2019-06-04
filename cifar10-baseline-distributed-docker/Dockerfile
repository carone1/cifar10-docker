# Use tensorflow as base image.
#FROM tensorflow/tensorflow:1.13.0rc0-gpu-py3
FROM tensorflow/tensorflow:1.13.1-gpu-py3

RUN apt-get update -y

ADD models models

# Install model dependencies.
RUN apt-get install -y python3-pip

ADD models/official/requirements.txt requirements.txt
RUN pip3 install --user -r requirements.txt

ENV PYTHONPATH="$PYTHONPATH:/models"

ADD models models
COPY cifar10_distributed_main.py models/official/resnet/
COPY resnet_run_loop_distributed.py models/official/resnet/

# Start 
CMD bash
