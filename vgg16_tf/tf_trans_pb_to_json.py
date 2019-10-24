#!/usr/bin/env python

from google.protobuf.json_format import MessageToJson

from tensorflow.core.framework import graph_pb2
graph_def = graph_pb2.GraphDef()

# to JSON
with open("vgg16_tf.pb", "rb") as pb:
    graph_def = pb
    tf_json = MessageToJson(graph_def)

    with open("vgg16_tf.json", "w") as output:
        output.write(tf_json)
