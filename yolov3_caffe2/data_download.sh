#!/bin/bash

# ONNXモデルダウンロード
echo "Download model..."
wget https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg16/vgg16.onnx
# テスト用画像ダウンロード
echo "Download Input Data..."
wget https://s3.amazonaws.com/model-server/inputs/kitten.jpg

chmod 755 vgg16.onnx kitten.jpg synset.txt
