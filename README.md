# Bimodal Supervised Contrastive Learning

## Recommended Requirements
python==3.8
rdkit
torch == 1.9.1
torch_geometric == 2.0.1
transformers == 4.12.2
ogb

We run the code on GPU and the version of CUDA is 11.6

## Usage example
1. Uncomment the command line corresponding to the dataset you want to test in the shell script 'autorun.sh'
2. Run 'autorun.sh'.
```sh
./autorun.sh
```
It will train the model and predict on the test set. 
