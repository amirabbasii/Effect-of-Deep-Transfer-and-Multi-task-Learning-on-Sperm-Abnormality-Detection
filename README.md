# The Blessing of Deep Transfer and Multi task Learning on Sperm Abnormality Detection
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)<br/>

# Introduction
This repository includes codes and models of "The Blessing of Deep Transfer and Multi task Learning on Sperm Abnormality Detection" paper.
link: https://doi.org/10.1016/j.compbiomed.2020.104121
## Dataset
First you should download the MHSMA dataset using:
```
git clone https://github.com/soroushj/mhsma-dataset.git
```
## Usage
First of all,the configuration file should be setted.So open dmtl.txt or dtl.txt and set the setting you want.This files contains paramaters of the model that you are going to train.<br/>
- dtl.txt have only one line and contains paramaters to train a DTL model.<br/>

- dmtl.txt contains two lines:paramaters of stage 1 are kept in the first line of the file and paramaters of stage 2 are kept in the second line of the file.<br/>
  Some paramaters have an aray of three values that they keep the value of three labels.To set them,consider this sequence:[Acrosome,Vacoule,Head].

- To train a DTL model,use the following commands and arguments:<br />
```
python train.py -t dtl [-e epchos] [-label label]  [-model model] [-w file] 
```
Argumetns:
| Argument | Description
| :--- | :----------
-t | type of network(dtl or dmtl)
-e| number of epochs
-label | label(a,v or h)
-model | pre-trained model
-w | name of best weihgt file
--phase| You can use it to choose stage in DMTL(1 or 2)
--second_model|The base model for second stage of DMTL

# 1.Train
- To choose a pre-trained model, you can use one of the following models:<br/>

| model argument | Description
| :--- | :----------
vgg_19 | VGG 19
vgg_16| VGG 16
resnet_50| Resnet 50
resnet_101| Resnet 101
resnet_502| Resnet 502



- To train a DMTL model,use the following commands and arguments:<br />
```
python train.py -t dmtl [--phase phase] [-e epchos] [-label label] [-model model] [-w file]

```
Also you can use your own pre-trained model by using address of your model instead of the paramaters been told in the table above.
```
Example:
python train.py -t dmtl --phase 1 -e 100 -label a -model C:\model.h5 -w w.h5

```
# 2.K Fold
- To perform K Fold on a model,use "-k_fold True" argument.
```
python train.py -k_fold True [-t type] [-e epchos] [-label label] [-model model] [-w file]

```
# 3.Threshold Search
- To find a good threshold for your model,use the following code:
```
python threshold.py [-t type] [-addr model address] [-l label]

```

## Models
The CNN models that were introduced and evaluated in our research paper can be found in the v1.0 release of this repository.

