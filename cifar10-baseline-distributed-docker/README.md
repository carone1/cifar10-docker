### Deep learning with TensorFlow RESNET model using cifar10 dataset 

This repository contains a demo of deep learning with Distributed TensorFlow for the Resnet model 

-   [Prerequisite](#prerequisite)
-   [Initialization](#initialization)
-   [Build Container](#build-container) 
-   [Start training](#start-training)


## prerequisite

All steps must be executede on each machine in clusters.

You need to install docker and nvidia-docker where you will run the training example

To leverage GPU with docker container you need to install NVIDIA-Docker

https://chunml.github.io/ChunML.github.io/project/Installing-NVIDIA-Docker-On-Ubuntu-16.04/

## Initialization

Before you start a demo you need to initialize workspace. The initialization include:

* Downloading Cifar10 dataset. Requires tensorflow cpu version to be installed
* Downloading official TensorFlow models repository.


To initialize the workspace you can use a single command:

```bash
$ pip3 install tensorflow
$ . init.sh
```


## build-container 

When workspace is initialized you can build Apache Ignite Docker image (that includes TensorFlow 1.13.1-gpu) using the following command:
The convenient way to start Apache Ignite cluster is to use Docker Compose:

```bash
$ docker build -f Dockerfile -t cifar10_distributed_gpu:10.0 .
```

When Docker image is ready you can start it using:

```bash
$ docker run -ti --name=cifar10_distributed --net=host --rm --runtime=nvidia -v /tmp:/tmp cifar10_distributed_gpu:1.0.0 bash
```

## Start training

You can start training from within the cifar10_distributed container (task_index variable must be different on each host):

```
cd models/official/resnet/
python3 cifar10_distributed_main.py --data_dir /tmp/cifar10_data/cifar-10-batches-bin/ --num_gpus 2  --worker_hosts=100.80.248.191:2222,100.80.248.193:2222 --task_index=0
```

The training is started. Your current shell shows you the output of the client script. 

