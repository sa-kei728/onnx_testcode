#!/bin/bash

convert-onnx-to-caffe2 yolov3.onnx --output yolov3_net.pb --init-net-output yolov3_param.pb
