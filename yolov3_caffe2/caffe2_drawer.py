#!/usr/bin/env python

NET="vgg16_net.pb"
PARAM="vgg16_param.pb"

with open(NET) as pb_net:
    net = pb_net.read()
with open(PARAM) as pb_param:
    param = pb_param.read()

p = workspace.Predictor(net, param)

print(p)
