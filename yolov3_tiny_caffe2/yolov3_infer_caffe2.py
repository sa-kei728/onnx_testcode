#!/usr/bin/env python

import onnx
import caffe2.python.onnx.backend
import skimage
import skimage.io
import skimage.transform
import cv2
import argparse
import numpy as np
import scipy

#argparse
parser = argparse.ArgumentParser(
            prog='Predict by ONNX model',
            usage='DeepLearning by ONNX',
            description='Predict Option',
            add_help=True,
            )
parser.add_argument('-op', '--operator', required=True)
args = parser.parse_args()

# Normalization
def Normalization(x, mean, std):
    x_norm = x
    for shape in range(x.ndim):
        x_norm[shape] = (x[shape] - mean[shape])/std[shape]
    return x_norm

#VGG16 base input shape
width   = 224
height  = 224

# [Input]
# All pre-trained models expect input images normalized in the same way, 
# i.e. mini-batches of 3-channel RGB images of shape (N x 3 x H x W), 
# where N is the batch size, and H and W are expected to be at least 224. 
# The inference was done using jpeg image.
# [PreProcess]
# The images have to be loaded in to a range of [0, 1] 
# and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. 
# The transformation should preferrably happen at preprocessing. Check imagenet_preprocess.py for code.
if args.operator == "skimage":
    img = skimage.img_as_float(skimage.io.imread('kitten.jpg'))
    img = skimage.transform.resize(img, (width, height)).astype(np.float32)
    img = Normalization(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
elif args.operator == "opencv":
    img = cv2.imread('kitten.jpg').astype(np.float32)
    img /= 255
    img = cv2.resize(img, (width, height)).astype(np.float32)
    img = Normalization(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
else:
    assert False

img4run = img[..., [2, 1, 0]]   #RGB -> BGR
img4run = np.rollaxis(img4run, 2)   #HWC -> CHW
img4run = img4run[np.newaxis, :]

# Load the ONNX model
model = onnx.load('yolov3_tiny.onnx')

# Run the ONNX model with Caffe2
predict = caffe2.python.onnx.backend.run_model(model, [img4run])
predict = np.squeeze(predict["vgg0_dense2_fwd"])
predict = scipy.special.softmax(predict)
ranking = np.argsort(predict)[::-1]

with open('synset.txt') as f:
    synset = f.readlines()
    for i in range(5):
        print('{}: {}'.format(synset[ranking[i]].strip(), predict[ranking[i]]))
