# onnx_testcode
onnx code inference test<br>

## Environment
CPU: Core-i5[x86_64]<br>
OS: Ubuntu 16.04 LTS<br>
anaconda: anaconda3-2019.07<br>
onnx: 1.5.0<br>
pytorch: 1.4.0a0+91a260c<br>

## setup
### 1. install pyenv(ref <http://blog.algolab.jp/post/2016/08/21/pyenv-anaconda-ubuntu/>)<br>
```
$ git clone git://github.com/yyuu/pyenv.git ~/.pyenv
$ git clone https://github.com/yyuu/pyenv-pip-rehash.git ~/.pyenv/plugins/pyenv-pip-rehash
$ echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
$ echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
$ echo 'eval "$(pyenv init -)"' >> ~/.bashrc
$ source ~/.bashrc
```

### 2. install anaconda for pyenv
```1.4.0a0+91a260c

pyenv install -l | grep anaconda3
pyenv install anaconda3-2019.07
pyenv global anaconda3-2019.07
echo 'export PATH="$PYENV_ROOT/versions/anaconda3-2019.07/bin:$PATH"' >> ~/.bashrc
conda init bash
source ~/.bashrc
```

### 3. install python3.6 for anaconda
```
conda create -n py36 python=3.6 anaconda
conda activate py36
```

### 4. install package
```
conda install -c menpo opencv=3.4.2 cudatoolkit=9.0.0 cudnn=7.1.2
conda install scikit-learn
```

### 5. install onnx(ref <https://github.com/onnx/onnx/blob/master/README.md>)
```
conda install -c conda-forge onnx
conda install -c conda-forge protobuf numpy
conda install -c anaconda mxnet
```

### 6. install pytorch and caffe2(ref <https://github.com/pytorch/pytorch/blob/master/README.md>)
```
conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing
git clone --recursive https://github.com/pytorch/pytorch

cd pytorch
# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install
```

### License
MIT
