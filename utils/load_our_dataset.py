import os
from itertools import repeat
from typing import Callable

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
import networkx as nx
import torch
from ogb.utils.mol import smiles2graph
from ogb.utils.url import decide_download, download_url, extract_zip
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils.convert import to_networkx
from transformers import RobertaTokenizerFast


def getmorganfingerprint(mol:Mol):
    """get the ECCP fingerprint

    Args:
        mol (Mol): The molecule
    """
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, 2))


def getmaccsfingerprint(mol:Mol):
    """get the MACCS fingerprint

    Args:
        mol (Mol): The molecule
    """
    fp = AllChem.GetMACCSKeysFingerprint(mol)
    return [int(b) for b in fp.ToBitString()]


class PygOurDataset(InMemoryDataset):
    def __init__(
        self,
        root:str="dataset",
        phase:str="train",
        dataname:str="hiv",
        smiles2graph:Callable=smiles2graph,
        transform=None,
        pre_transform=None,
    ):
        """Load our dataset. 

        Args:
            root (str, optional): The local position of the dataset. Defaults to "dataset".
            phase (str, optional): The data is train, validation or test set. Defaults to "train".
            dataname (str, optional): The name of the dataset. Defaults to "hiv".
            smiles2graph (Callable, optional): Generate the molecular graph from the SMILES string. Defaults to smiles2graph.
        """

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = os.path.join(root, dataname)
        self.version = 1
        self.dataname = dataname
        self.phase = phase
        self.aug = "none"
        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            "seyonec/ChemBERTa_zinc250k_v2_40k", max_len=100
        )

        super(PygOurDataset, self).__init__(self.folder, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return self.phase + "_" + self.dataname + ".csv"

    @property
    def processed_file_names(self):
        return self.phase + "_" + self.dataname + ".pt"

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
        else:
            print("Stop download.")
            exit(-1)

    def process(self):
        data_df = pd.read_csv(
            os.path.join(self.raw_dir, self.phase + "_" + self.dataname + ".csv")
        )
        smiles_list = data_df["smiles"]
        homolumogap_list = data_df[data_df.columns.difference(["smiles", "mol_id", "num", "name"])]

        encodings = self.tokenizer(smiles_list.tolist(), truncation=True, padding=True)

        print("Converting SMILES strings into graphs...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            homolumogap = homolumogap_list.iloc[i]
            # homolumogap_list[i]
            graph = self.smiles2graph(smiles)

            rdkit_mol = AllChem.MolFromSmiles(smiles)
            mgf = getmorganfingerprint(rdkit_mol)
            maccs = getmaccsfingerprint(rdkit_mol)

            assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
            assert len(graph["node_feat"]) == graph["num_nodes"]

            data.__num_nodes__ = int(graph["num_nodes"])
            data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
            data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
            data.y = torch.Tensor([homolumogap])
            data.input_ids = torch.Tensor(encodings.input_ids[i])
            data.attention_mask = torch.Tensor(encodings.attention_mask[i])
            data.mgf = torch.tensor(mgf)
            data.maccs = torch.tensor(maccs)
            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        return data