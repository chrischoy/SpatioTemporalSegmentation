# Spatio-Temporal Segmentation

This repository contains the accompanying code for [4D-SpatioTemporal ConvNets: Minkowski Convolutional Neural Networks, CVPR'19](https://arxiv.org/abs/1904.08755).


## ScanNet Training

1. Download the ScanNet dataset from [the official website](http://kaldir.vc.in.tum.de/scannet_benchmark/documentation). You need to sign the terms of use.

2. Next, preprocess all scannet raw point cloud with the following command after you set the path correctly.

```
python -m lib.datasets.prepreocessing.scannet
```

3. Train the network with

```
export BATCH_SIZE=N;
./scripts/train_scannet.sh 0 \
	-default \
	"--scannet_path /path/to/preprocessed/scannet"
```

Modify the `BATCH_SIZE` accordingly.

The first argument is the GPU id and the second argument is the path postfix
and the last argument is the miscellaneous arguments.

The official evaluation metric for ScanNet is mIoU. OA, Overal Accuracy is not the official metric similar to the 2D image semantic segmentation as it is a lot easier and doesn't reflect the quality of the semantic segmentation. This is due to the fact that most of the scenes consist of large structures such as walls, floors and these will dominate the statistics. Thus, overall accuracy should not be used solely to prevent reviewers from comparing the baselines fairly.

## Synthia 4D Experiment

1. Download the dataset from [download](http://cvgl.stanford.edu/data2/Synthia4D.tar)

2. Extract

```
cd /path/to/extract/synthia4d
wget http://cvgl.stanford.edu/data2/Synthia4D.tar
tar -xf Synthia4D.tar
tar -xvjf *.tar.bz2
```

3. Training

```
export BATCH_SIZE=N; \
./scripts/train_synthia4d.sh 0 \
	"-default" \
	"--synthia_path /path/to/extract/synthia4d"
```

The above script trains a network. You have to change the arguments accordingly. The first argument to the script is the GPU id. Second argument is the log directory postfix; change to mark your experimental setup. The final argument is a series of the miscellaneous aruments. You have to specify the synthia directory here. Also, you have to wrap all arguments with " ".


## Model Zoo

| Model         | Dataset | Voxel Size | Performance              | Link   |
|:-------------:|:-------:|:----------:|:------------------------:|:------:|
| Mink16UNet34C | ScanNet | 2cm        | Test set 73.6% mIoU      | [download](https://node1.chrischoy.org/data/publications/minknet/Mink16UNet34C_ScanNet.pth) |
