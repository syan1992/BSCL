import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaConfig, RobertaTokenizerFast, Trainer, TrainingArguments
from transformers import PreTrainedModel, RobertaModel
import random
class SMILESBert(nn.Module):
    def __init__(self, num_tasks=1):
        super(SMILESBert,self).__init__()
        dim_in = 512 
        feat_dim = 128
        self.bertConfig=RobertaConfig.from_pretrained('seyonec/ChemBERTa_zinc250k_v2_40k')
        self.roberta = RobertaModel(self.bertConfig, add_pooling_layer=False)
        self.model = RobertaModel.from_pretrained('seyonec/ChemBERTa_zinc250k_v2_40k', config=self.bertConfig)
        self.dense = nn.Linear(self.bertConfig.hidden_size, 128)
        self.dropout = nn.Dropout(0.5)
        
        
    def forward(self,ids,mask, phase='train'):
        #with torch.no_grad():
        features =self.model(input_ids=ids,attention_mask=mask)[0]
        feat = features[:,0,:]
        
        feat = self.dropout(feat)
        feat = self.dense(feat)
        
        if phase=='train':
            return feat
        else:
            return feat
