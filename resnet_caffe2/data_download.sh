#!/bin/bash

# ONNXモデルダウンロード
echo "Download model..."
wget https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v2/resnet50v2.onnx
# テスト用画像ダウンロード
echo "Download Input Data..."
wget https://s3.amazonaws.com/model-server/inputs/kitten.jpg
# テスト用ラベルダウンロード
echo "Download Label..."
wget https://s3.amazonaws.com/onnx-model-zoo/synset.txt

chmod 755 resnet50v2.onnx kitten.jpg synset.txt
