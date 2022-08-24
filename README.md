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
We list all command lines in the shell script 'autorun.sh' for the seven datasets we test in our experiments. 
1. Uncomment the command line corresponding to the dataset you want to test in 'autorun.sh'
2. Run 'autorun.sh'.
```sh
./autorun.sh
```
We save the model with the best performance on the validation set and evaluate the best model with the test set.
Both model and test results will be saved in the 'save' folder.

## Acknowledgement
Supervised contrastive learning : https://github.com/HobbitLong/SupContrast  
Deepgcn : https://github.com/lightaime/deep_gcns_torch
