#!/bin/bash

convert-onnx-to-caffe2 vgg16.onnx --output vgg16_net.pb --init-net-output vgg16_param.pb
