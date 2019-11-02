# Spatio-Temporal Segmentation

This repository contains the accompanying code for [4D-SpatioTemporal ConvNets: Minkowski Convolutional Neural Networks, CVPR'19](https://arxiv.org/abs/1904.08755).


## ScanNet Training

1. First, preprocess all scannet raw point cloud with the following command after you set the path correctly.

```
python -m lib.datasets.prepreocessing.scannet
```

2. Download the v2 official splits from [https://github.com/ScanNet/ScanNet/tree/master/Tasks/Benchmark](https://github.com/ScanNet/ScanNet/tree/master/Tasks/Benchmark) and save them to the scannet preprocessed root directory.

3. Create `scannetv2_trainval.txt` by concatenating `scannetv2_train.txt` and `scannetv2_val.txt`.

4. Train the network with

```
export BATCH_SIZE=N; \
	./scripts/train_scannet.sh 0 \
	-default \
	"--scannet_path /path/to/preprocessed/scannet"
```

The first argument is the GPU id and the second argument is the path postfix
and the last argument is the miscellaneous arguments.


## Synthia 4D Experiment

1. Download the dataset from [download](http://cvgl.stanford.edu/data2/Synthia4D.tar)

2. Extract

```
tar -xf Synthia4D.tar
tar -xzf *.tar.bz2
```


## Model Zoo

| Model         | Dataset | Voxel Size | Performance              | Link   |
|:-------------:|:-------:|:----------:|:------------------------:|:------:|
| Mink16UNet34C | ScanNet | 2cm        | Test set 73.6% mIoU      | [download](https://node1.chrischoy.org/data/publications/minknet/Mink16UNet34C_ScanNet.pth) |
