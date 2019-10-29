#!/usr/bin/env python

import onnx
import warnings
warnings.filterwarnings('ignore') # Ignore all the warning messages in this tutorial
import mxnet as mx
from mxnet.contrib.onnx.onnx2mx.import_model import import_model
from mxnet.context import cpu
import mxnet.ndarray as nd
import json

# インストールするonnxファイル
model_path= 'vgg16.onnx'
sym, arg_params, aux_params = import_model(model_path)

# symbol save to json
sym.save('vgg16-symbol.json')

# save param to dict file
save_dict = {('arg:%s' % k) : v.as_in_context(cpu()) for k, v in arg_params.items()}
save_dict.update({('aux:%s' % k) : v.as_in_context(cpu()) for k, v in aux_params.items()})
param_name = '%s-%04d.params' % ("vgg16", 0)
nd.save(param_name, save_dict)
