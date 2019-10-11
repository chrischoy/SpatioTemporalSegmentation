# Spatio-Temporal Segmentation

This repository contains the accompanying code for [4D-SpatioTemporal ConvNets: Minkowski Convolutional Neural Networks, CVPR'19](https://arxiv.org/abs/1904.08755).


## ScanNet Training

First, preprocess all scannet raw point cloud with the following command after you set the path correctly.

```
python -m lib.datasets.prepreocessing.scannet
```

Then, train the scannet network with

```
./scripts/train_scannet.sh 0 -default "--scannet_path /path/to/preprocessed/scannet"
```

The first argument is the GPU id and the second argument is the path postfix
and the last argument is the miscellaneous arguments.

