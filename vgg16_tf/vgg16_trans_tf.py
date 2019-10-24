#!/usr/bin/env python

import onnx
import warnings
warnings.filterwarnings('ignore') # Ignore all the warning messages in this tutorial
from onnx_tf.backend import prepare
import tensorflow as tf

model = onnx.load('vgg16.onnx') # Load the ONNX file
tf_rep = prepare(model) # Import the ONNX model to Tensorflow
print(tf_rep.inputs)    # input node
print(tf_rep.outputs)   # output node
print(tf_rep.tensor_dict) # All nodes in the model
tf_rep.export_graph("vgg16_tf.pb")  #Export Tensorflow model_graph (Protocol Buffer)
