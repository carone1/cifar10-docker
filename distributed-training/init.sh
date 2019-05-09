
if [ -f ./apache-ignite-2.7.0-bin.zip ] ; then
    echo "Apache Ignite already downloaded..."
else
    wget http://mirrors.advancedhosters.com/apache//ignite/2.7.0/apache-ignite-2.7.0-bin.zip 
fi


if [ -d ./apache-ignite-2.7.0-bin ]
then
    echo "Apache Ignite already in place ..."
else
    echo "Unzipping Apache Ignite package..."
    unzip -qo apache-ignite-2.7.0-bin.zip
    #rm apache-ignite-2.7.0-SNAPSHOT-bin.zip
fi

echo "Updating path..."
cd apache-ignite-2.7.0-bin/bin
export PATH=`pwd`:$PATH
cd ../..



echo "Downloading cifar10..."
python3 cifar10_download_and_extract.py

if [ -d models ] ; then
    echo "models is already present"
else
    echo "Downloading models..."
    git clone https://github.com/tensorflow/models.git --depth 1 --branch r1.13.0
    rm -rf models/research models/samples models/tutorials models/.git
    patch -p0 < models.diff
fi

echo "Initialization completed"

