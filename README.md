# Feature Inference Attack on Model Predictions in Vertical Federated Learning

## Overview
Feature Inference Attack on Model Predictions in Vertical Federated Learning[1] investigates the potential privacy leakages in the model prediction stage of vertical FL. This work consists of three attack methods: `Equation Solving Attack` (ESA) on the logistic regression (LR) models, `Path Restriction Attack` (PRA) on the decision tree (DT) models, and `Generative Regression Network Attack` (GRNA) on the Logistic Regression, Random Forest (RF) and Neural Network (NN) models.



## Citation
If you use our results or this codebase in your research, then please cite this paper:
```
@article{luo2020feature,
  title={Feature Inference Attack on Model Predictions in Vertical Federated Learning},
  author={Luo, Xinjian and Wu, Yuncheng and Xiao, Xiaokui and Ooi, Beng Chin},
  journal={arXiv preprint arXiv:2010.10152},
  year={2020}
}
```

## How to run
### Install dependencies
This code is written in python3.
To run it, you need to install `numpy`, `pytorch` and `sklearn` first.

### Important script arguments
Before running GRNA, you can configure the parameters of generator, datasets and global classifier in `config.ini`. The key names are self-explanatory.

  
### Equation Solving Attack:
ESA is applicable to the LR models. 
To initiate an ESA attack, run the following script:
```
python main-esa.py
```

### Path Restriction Attack:
PRA is applicable to the DT models. 
To initiate a PRA attack, run the following script:
```
python main-pra.py
```


### Generative Regression Network Attack:
GRNA is applicable to the LR, RF and NN models.
To initiate a GRNA attack, run the following script:
```
python main-grna.py
```


## Reference:
[1] [**Feature Inference Attack on Model Predictions in Vertical Federated Learning**](https://arxiv.org/abs/2010.10152), *Xinjian Luo, Yuncheng Wu, Xiaokui Xiao, Beng Chin Ooi*, ICDE 2021.
