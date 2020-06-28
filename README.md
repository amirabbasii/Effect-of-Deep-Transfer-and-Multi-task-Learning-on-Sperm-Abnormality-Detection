# The Blessing of Deep Transfer and Multi task Learning on Sperm Abnormality Detection
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
This repository includes codes and models of "The Blessing of Deep Transfer and Multi task Learning on Sperm Abnormality Detection" paper.
## Contents
- [Usage](#Usage)


## Usage
- To train a DTL model,use the following commands and arguments:<br />
```
python train.py -t dtl [-e epchos] [--label label]  [-model model]
```

- To train a DMTL model,use the following commands and arguments:<br />
```
python train.py -t dmtl [-e epchos] [--label label] [--phase phase] [-model model]
```


