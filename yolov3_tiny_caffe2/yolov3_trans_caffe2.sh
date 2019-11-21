#!/bin/bash

convert-onnx-to-caffe2 yolov3_tiny.onnx --output yolov3_tiny_net.pb --init-net-output yolov3_tiny_param.pb
