# Feature Inference Attack on Model Predictions in Vertical Federated Learning

## Overview
Feature Inference Attack on Model Predictions in Vertical Federated Learning [1] investigates the potential privacy leakages in the model prediction stage of vertical FL. This work consists of three attack methods: `Equation Solving Attack` (ESA) on the logistic regression (LR) models, `Path Restriction Attack` (PRA) on the decision tree (DT) models, and `Generative Regression Network Attack` (GRNA) on the Logistic Regression, Random Forest (RF) and Neural Network (NN) models.


## Citation
If you use our results or this codebase in your research, then please cite this paper:
```
@inproceedings{luo2021feature,
  title={Feature inference attack on model predictions in vertical federated learning},
  author={Luo, Xinjian and Wu, Yuncheng and Xiao, Xiaokui and Ooi, Beng Chin},
  booktitle={2021 IEEE 37th International Conference on Data Engineering (ICDE)},
  pages={181--192},
  year={2021},
  organization={IEEE}
}

```

## How to run
### Install dependencies
This code is written in python3.
To run it, you need to install `numpy`, `pytorch` and `sklearn` first.


### Important script arguments
Before running these attacks, you can configure the parameters of generator, datasets and global classifier in `config.ini`. The key names in this configure file are self-explanatory.

  
### Equation Solving Attack
ESA is applicable to the LR models. 
To initiate an ESA attack, run the following script:
```
cd ESA/
python main-esa.py
```

### Path Restriction Attack
PRA is applicable to the DT models. 
To initiate a PRA attack, run the following script:
```
cd PRA/
python main-pra.py
```

### Generative Regression Network Attack
GRNA is applicable to the LR, RF and NN models.
To initiate a GRNA attack, run the following script:
```
cd GRNA/
python main-grna.py
```

## Reference
[1] [**Feature Inference Attack on Model Predictions in Vertical Federated Learning**](https://arxiv.org/abs/2010.10152), *Xinjian Luo, Yuncheng Wu, Xiaokui Xiao, Beng Chin Ooi*, ICDE 2021.
