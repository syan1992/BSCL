from torch import Tensor
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel


class SMILESBert(nn.Module):
    """SMILES branch"""
    def __init__(self):
        super(SMILESBert, self).__init__()
        self.bertConfig = RobertaConfig.from_pretrained(
            "seyonec/ChemBERTa_zinc250k_v2_40k")
        self.roberta = RobertaModel(self.bertConfig, add_pooling_layer=False)
        self.model = RobertaModel.from_pretrained(
            "seyonec/ChemBERTa_zinc250k_v2_40k", config=self.bertConfig
        )
        self.dense = nn.Linear(self.bertConfig.hidden_size, 128)
        self.dropout = nn.Dropout(0.5)

    def forward(self, ids:Tensor, mask:Tensor):
        """Generate the embedding of the SMILES branch. 

        Args:
            ids (Tensor): Encoding of the SMILES strings. 
            mask (Tensor): Attention mask. 

        Returns:
            The embedding of the SMILES branch. 
        """
        features = self.model(input_ids=ids, attention_mask=mask)[0]
        feat = features[:, 0, :]

        feat = self.dropout(feat)
        feat = self.dense(feat)

        return feat
