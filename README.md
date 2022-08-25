# Molecular Property Prediction based on Bimodal Supervised Contrastive Learning

## Recommended Requirements
The scripts are tested under python 3.8 with the following packages, 
```
matplotlib==3.4.3  
networks==0.3.7   
numpy==1.21.2  
ogb==1.3.2  
pandas==1.3.3  
rdkit==2022.3.5  
rdkit_pypi==2021.9.4  
scikit_learn==1.1.2  
scipy==1.7.1    
torch==1.9.1  
torch_geometric==2.0.1  
torch_scatter==2.0.8  
torchvision==0.10.1  
tqdm==4.50.0  
transformers==4.12.2  
```
We run the code on GPU and the version of CUDA is 11.6

## Datasets
Please find the 'datasets' folder for the example of the data. The data should be split into train/validation/test subsets at first. 

## Usage example
We list all command lines in the shell script 'autorun.sh' for the seven datasets (freesolv, delaney, lipophilicity, bace, sider, tox21, clintox) we test in our experiments. 
Run 'autorun.sh' with the name of the dataset as a parameter.
```sh
./autorun.sh freesolv
```
We save the model with the best performance on the validation set and evaluate the best model with the test set.
Both model and test results will be saved in the 'save' folder.

## Hyperparameters
Some specific hyperparameters in this work,  
|  Name   | Description  |
| :---        |    :----:   |
|  wscl  | The weight of the supervised contrastive loss in the loss function. Suggest to test values in [0.1 to 1]|
| wrecon  | The weight of the reconstruction loss in the loss function. Suggest to test values in [0.1 to 1]|
| gamma1  | The hyperparameter of the weighted supervised contrastive loss for the regression task. Suggest to test values in [2,3,4] |
| gamma2  | The hyperparameter of the weighted supervised contrastive loss for the regression task. Suggest to test values in [1,2,3] |

## Acknowledgement
Supervised contrastive learning : https://github.com/HobbitLong/SupContrast  
Deepgcn : https://github.com/lightaime/deep_gcns_torch
