#!/usr/bin/env python

NET="resnet50v2_net.pb"
PARAM="resnet50v2_param.pb"

with open(NET, 'rb', encoding='utf-8') as pb_net:
    net = pb_net.read()
with open(PARAM, 'rb', encoding='utf-8') as pb_param:
    param = pb_param.read()

p = workspace.Predictor(net, param)

print(p)
