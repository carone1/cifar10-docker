### Deep learning with TensorFlow RESNET model using cifar10 dataset 

This repository contains a demo of deep learning with TensorFlow for the Resnet model 

-   [Prerequisite](#prerequisite)
-   [Initialization](#initialization)
-   [Build Container](#build-container) 
-   [Start training](#start-training)


## prerequisite

You need to install docker and nvidia-docker where you will run the trainingi example

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

To clean the workspace:

```bash
$ . clear.sh
```

## build-container 

When workspace is initialized you can build Apache Ignite Docker image (that includes TensorFlow 1.13.1-gpu) using the following command:
The convenient way to start Apache Ignite cluster is to use Docker Compose:

```bash
$ docker build -f Dockerfile_for_gpu -t cifar10_gpu:10.0 .
```

When Docker image is ready you can start it using:

```bash
$ docker run -ti -rm -name cifar10 -v /tmp:/tmp cifar_gpu:1.0.0
```

## Start training

You can start training:

```
cd models/official/resnet/
python3 cifar10_main.py --data_dir /tmp/cifar10_data/cifar-10-batches-bin/ --num_gpus 4
```

The training is started. Your current tab shows you the output of the client script. 

