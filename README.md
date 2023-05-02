# C3N：Content-Constrained Convolutional Network for Mural Image Completion

### [Paper](https://link.springer.com/article/10.1007/s00521-022-07806-0)

## Requirements
numpy==1.14.4

Pillow==5.1.0

six==1.11.0

tensorboardX==1.2

torch==0.4.1

torchvision==0.2.1

tqdm==4.23.4

## Preparation works
To generate binary masks, use
```
python generate_data.py
```

To generate the image covered by the mask, that is, generate the simulated damaged image, use
```
python 1test.py
```

## Training and testing
To conduct network model training, use
```
python train.py
```
The image data set and mask data set can be simply modified at the beginning of the code as required.

To generate a repair image, use
```
python 2test.py
```

## Citation

If you find our code or paper useful, please cite the paper:
```bash
@article{PengWZ23,
title = {C3N: Content-constrained convolutional network for mural image completion},
author = {Xianlin Peng, Huayu Zhao, Xiaoyu Wang, Yongqin Zhang, Zhan Li, Qunxi Zhang, Jun Wang, Jinye Peng, Haida Liang},
journal = {Neural Computing and Applications},
volume = {35},
article id = {},
pages = {1959-1970},
year={2023}
}
```
