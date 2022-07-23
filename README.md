# FIFO: Learning Fog-invariant Features for Foggy Scene Segmentation

### [Project Page](http://cvlab.postech.ac.kr/research/FIFO/) | [Paper](https://arxiv.org/abs/2204.01587)
This repo is the official implementation of [**CVPR 2022 Oral, Best Paper Finalist**] paper: "[**FIFO**: Learning Fog-invariant Features for Foggy Scene Segmentation](https://arxiv.org/abs/2204.01587)".

> [FIFO: Learning Fog-invariant Features for Foggy Scene Segmentation](https://arxiv.org/abs/2204.01587)     
> [Sohyun Lee](https://sohyun-l.github.io)<sup>1</sup>, Taeyoung Son<sup>2</sup>, [Suha Kwak](http://cvlab.postech.ac.kr/~suhakwak/)<sup>1</sup>\
> POSTECH<sup>1</sup>, NALBI<sup>2</sup>\
> accept to CVPR 2022 as an oral presentation 

![Overall_architecture](https://user-images.githubusercontent.com/57887512/161761968-436766d9-363d-463d-b8b3-f48ba2a2a949.png)


## Overview
Robust visual recognition under adverse weather conditions is of great importance in real-world applications. In this context, we propose a new method for learning semantic segmentation models robust against fog. Its key idea is to consider the fog condition of an image as its style and close the gap between images with different fog conditions in neural style spaces of a segmentation model. In particular, since the neural style of an image is in general affected by other factors as well as fog, we introduce a fog-pass filter module that learns to extract a fog-relevant factor from the style. Optimizing the fog-pass filter and the segmentation model alternately gradually closes the style gap between different fog conditions and allows to learn fog-invariant features in consequence. Our method substantially outperforms previous work on three real foggy image datasets. Moreover, it improves performance on both foggy and clear weather images, while existing methods often degrade performance on clear scenes.

## Citation
If you find our code or paper useful, please consider citing our paper:

```BibTeX
@inproceedings{lee2022fifo,
  author    = {Sohyun Lee and Taeyoung Son and Suha Kwak},
  title     = {FIFO: Learning Fog-invariant Features for Foggy Scene Segmentation},
  booktitle = {Proceedings of the {IEEE/CVF} Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022}
}
```

## Experimental Results
![Main_qual](https://user-images.githubusercontent.com/57887512/163107476-7e70cebe-6b38-497f-b5bd-f8d6979a8fb0.png)


## Dataset
+ **Cityscapes**: Download the [Cityscapes Dataset](https://www.cityscapes-dataset.com/), and put it in the /root/data1/Cityscapes folder

+ **Foggy Cityscapes**: Download the [Foggy Cityscapes Dataset](https://www.cityscapes-dataset.com/), and put it in the /root/data1/leftImg8bit_foggyDBF folder

+ **Foggy Zurich**: Download the [Foggy Zurich Dataset](https://people.ee.ethz.ch/~csakarid/Model_adaptation_SFSU_dense/), and put it in the /root/data1/Foggy_Zurich folder

+ **Foggy Driving and Foggy Driving Dense**: Download the [Foggy Driving Dataset](https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/), and put it in the /root/data1/Foggy_Driving folder

## Installation
This repository is developed and tested on

- Ubuntu 16.04
- Conda 4.9.2
- CUDA 11.4
- Python 3.7.7
- PyTorch 1.5.0

## Environment Setup
* Required environment is presented in the 'FIFO.yaml' file
* Clone this repo
```bash
~$ git clone https://github.com/sohyun-l/fifo
~$ cd fifo
~/fifo$ conda env create --file FIFO.yaml
~/fifo$ conda activate FIFO.yaml
```

## Pretrained Models
PRETRAINED_SEG_MODEL_PATH = '[./Cityscapes_pretrained_model.pth](https://drive.google.com/file/d/1IKBXXVhYfc6n5Pw23g7HsH_QzqOG03c6/view?usp=sharing)'


PRETRAINED_FILTER_PATH = '[./FogPassFilter_pretrained.pth](https://drive.google.com/file/d/1xHkL3Y8Y5sHoGkmcevrfMdhFxafVF4_G/view?usp=sharing)' 


## Testing
BEST_MODEL_PATH = '[./FIFO_final_model.pth](https://drive.google.com/file/d/1UF-uotKznN_wqqNqwIkPnpw55l8T9b62/view?usp=sharing
)'

Evaluating FIFO model
```bash
(fifo) ~/fifo$ python evaluate.py --file-name 'FIFO_model' --restore-from BEST_MODEL_PATH
```


## Training
Pretraining fog-pass filtering module
```bash
(fifo) ~/fifo$ python main.py --file-name 'fog_pass_filtering_module' --restore-from PRETRAINED_SEG_MODEL_PATH --modeltrain 'no'
```
Training FIFO
```bash
(fifo) ~/fifo$ python main.py --file-name 'FIFO_model' --restore-from PRETRAINED_SEG_MODEL_PATH --restore-from-fogpass PRETRAINED_FILTER_PATH --modeltrain 'train'
```


## Acknowledgments
Our code is based on [AdaptSegNet](https://github.com/wasidennis/AdaptSegNet), [RefineNet-lw](https://github.com/DrSleep/light-weight-refinenet), and [Pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning).
We also thank [Christos Sakaridis](http://people.ee.ethz.ch/~csakarid/) for sharing [datasets](http://people.ee.ethz.ch/~csakarid/Model_adaptation_SFSU_dense/) and code of [CMAda](https://arxiv.org/pdf/1901.01415.pdf).
If you use our model, please consider citing them as well.
