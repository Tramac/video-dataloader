# Video DataLoader
Some dataloader for video datasets.

| datasets | videos | classes |
| :------: | :----: | :-----: |
|  UCF101  | 13320  |   101   |

## Introduction

This repo contains several dataloader for video datasets, including UCF101, your own datasets.

## Datasets

- UCF101 [download](https://www.crcv.ucf.edu/data/UCF101.php)

Make sure to put files as following structure:

```shell
UCF101
├── ApplyEyeMakeup
│   ├── v_ApplyEyeMakeup_g01_c01.avi
│   └── ...
├── ApplyLipstick
│   ├── v_ApplyLipstick_g01_c01.avi
│   └── ...
└── Archery
│   ├── v_Archery_g01_c01.avi
│   └── ...
```

After pre-processing, the output dir's structure is as follows:

```
ucf101
├── ApplyEyeMakeup
│   ├── v_ApplyEyeMakeup_g01_c01
│   │   ├── 00001.jpg
│   │   └── ...
│   └── ...
├── ApplyLipstick
│   ├── v_ApplyLipstick_g01_c01
│   │   ├── 00001.jpg
│   │   └── ...
│   └── ...
└── Archery
│   ├── v_Archery_g01_c01
│   │   ├── 00001.jpg
│   │   └── ...
│   └── ...
```

## TODO

- [ ] Split train/val/test

- [ ] More handful

- [ ] More dataloader
- [ ] Elegant code

## WeChat

Considering whether it is necessary.