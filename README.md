# The Blessing of Deep Transfer and Multi task Learning on Sperm Abnormality Detection
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)<br/>

# Introduction
This repository includes codes and models of "The Blessing of Deep Transfer and Multi task Learning on Sperm Abnormality Detection" paper.

## Usage

| Argument | Description
| :--- | :----------
-t | type of network
-e| number of epochs
--label | label
--model | pre-trained model


- To choose a pre-trained model, you can use one of the following models:<br/>

| model argument | Description
| :--- | :----------
vgg_19 | VGG 19
vgg_16| VGG 16
densenet_121 | Densenet 121
resnet_50| Resnet 50

- To train a DTL model,use the following commands and arguments:<br />
```
python train.py -t dtl [-e epchos] [--label label]  [-model model]
```

- To train a DMTL model,use the following commands and arguments:<br />
```
python train.py -t dmtl [-e epchos] [--label label] [--phase phase] [-model model]
```

## Models
The CNN models that were introduced and evaluated in our research paper can be found in the v1.0 release of this repository.
