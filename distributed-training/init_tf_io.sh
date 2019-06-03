if [ -d ./ignite ]
then
    echo "Apache Ignite already downloaded..."
else
    echo "Downloading Apache Ignite..."
    git clone https://github.com/apache/ignite.git
    cd ignite
    #git checkout ea33ec7f0af8fcad113cd92953fba0e8e5502dfa
    # grabbing ignite 2.7.5rc3
    #git checkout b141a3eb7487d396bee7919533019f941a504b17

    # grabbing latest from master branch
    git checkout 8e69ae7648f50aeab7884ae58624be0a632cdd31 
    cd ..
fi

if [ -d ./apache-ignite-2.8.0-SNAPSHOT-bin ]
then
    echo "Apache Ignite already built..."
else
    #echo "Patching ignite for tensorflow-io"
    # switch to tensorflow-io
    patch -p0 < ignite.diff

    echo "Building Apache Ignite..."
    cd ignite
    mvn clean package -q -B -DskipTests -Prelease

    echo "Unzipping Apache Ignite package..."
    cd ..
    mv ignite/target/bin/apache-ignite-2.8.0-SNAPSHOT-bin.zip ./
    unzip -qo apache-ignite-2.8.0-SNAPSHOT-bin.zip
    rm apache-ignite-2.8.0-SNAPSHOT-bin.zip
fi



echo "Updating path..."
cd apache-ignite-2.8.0-SNAPSHOT-bin/bin
export PATH=`pwd`:$PATH
cd ../..

echo "Downloading cifar10..."
python3 cifar10_download_and_extract.py

echo "Downloading models..."
rm -rf models
#git clone https://github.com/tensorflow/models.git --depth 1 --branch r1.13.0
# downloading the zip file directly is faster
wget https://github.com/tensorflow/models/archive/v1.13.0.zip
unzip -qo v1.13.0.zip
mv models-1.13.0 models
rm -rf models/research models/samples models/tutorials models/.git
# regular patch were tensorflow contrib is used 
#patch -p0 < models.diff
# patch where tensorflow io module is used
patch -p0 < models_tf_io.diff

echo "Initialization completed"
