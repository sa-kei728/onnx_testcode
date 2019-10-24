#!/bin/bash

# trans to caffe.pb
mmtoir -f tensorflow -w vgg16_tf.pb --inNodeName data --inputShape 3,224,224 --dstNodeName vgg0_dense2_fwd -o vgg16

#create python code and npy for caffe
mmtocode -f caffe -n vgg16.pb -w vgg16.npy -o caffe_vgg16.py -ow caffe_vgg16.npy

#create prototxt and caffemodel by python code and npy
mmtomodel -f caffe -in caffe_vgg16.py -iw caffe_vgg16.npy -o mmdnn_vgg16
