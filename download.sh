#!/bin/bash

rm -rf pretrained
mkdir pretrained
mkdir pretrained/inception_resnet_v2
mkdir pretrained/resnet_v2_152

cd pretrained/inception_resnet_v2
echo 'Downloading inception_resnet_v2 checkpoint ...'
curl -L 'http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz' -o checkpoint.tar.gz
echo 'Extracting ...'
tar -xzf checkpoint.tar.gz
rm checkpoint.tar.gz
mv inception_resnet_v2_2016_08_30.ckpt inception_resnet_v2.ckpt

cd ../resnet_v2_152
echo 'Downloading resnet_v2_152 checkpoint ...'
curl -L 'http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz' -o checkpoint.tar.gz
echo 'Extracting ...'
tar -xzf checkpoint.tar.gz
rm checkpoint.tar.gz

echo 'Done!'
