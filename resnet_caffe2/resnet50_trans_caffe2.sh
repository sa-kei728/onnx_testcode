#!/bin/bash

convert-onnx-to-caffe2 resnet50v2.onnx --output resnet50v2_net.pb --init-net-output resnet50v2_param.pb
