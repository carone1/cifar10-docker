
echo "Downloading models..."
rm -rf models
git clone https://github.com/tensorflow/models.git --depth 1 --branch r1.13.0
rm -rf models/research models/samples models/tutorials models/.git

echo "Downloading cifar10..."
python3 ./models/official/resnet/cifar10_download_and_extract.py

echo "Initialization completed"
