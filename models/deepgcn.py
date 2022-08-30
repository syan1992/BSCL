import numpy as np
from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, Set2Set

from models.deepgcn_vertex import GENConv
from models.deepgcn_nn import AtomEncoder, BondEncoder, MLP, norm_layer


class DeeperGCN(torch.nn.Module):
    """DeeperGCN network."""

    def __init__(
        self,
        num_gc_layers: int,
        dropout: float,
        block: str,
        conv_encode_edge: bool,
        add_virtual_node: bool,
        hidden_channels: int,
        num_tasks: int,
        aggr: str = "add",
        graph_pooling: str = "mean",
        t: float = 1.0,
        learn_t: bool = False,
        p: float = 1.0,
        learn_p: bool = False,
        y: float = 0.0,
        learn_y: bool = False,
        mlp_layers: int = 1,
        norm: str = "batch",
    ):
        """
        Args:
            num_gc_layers (int): Depth of the network.
            dropout (float): Dropout rate.
            block (str): Selection of the block, res, res+ or plain.
            add_virtual_node (bool): Whether add virtual node.
            hidden_channels (int): Number of hidden channels.
            num_tasks (int): Number of tasks.
            aggr (str, optional): Selection of aggregation methods. add, sum or max. Defaults to "add".
            mlp_layers (int, optional): Number of MLP layers. Defaults to 1.
            norm (str, optional): Selection of the normalization methods. batch or layer. Defaults to "batch".
        """
        super(DeeperGCN, self).__init__()

        self.num_gc_layers = num_gc_layers
        self.gcns = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        self.dropout = dropout
        self.block = block
        self.conv_encode_edge = conv_encode_edge
        self.add_virtual_node = add_virtual_node

        hidden_channels = hidden_channels
        num_tasks = num_tasks
        aggr = aggr

        t = 1.0
        self.learn_t = learn_t
        p = 1.0
        self.learn_p = learn_p
        y = 0.0
        self.learn_y = learn_y

        self.msg_norm = False
        learn_msg_scale = False

        mlp_layers = mlp_layers

        print(
            "The number of layers {}".format(self.num_gc_layers),
            "Aggr aggregation method {}".format(aggr),
            "block: {}".format(self.block),
        )
        if self.block == "res+":
            print("LN/BN->ReLU->GraphConv->Res")
        elif self.block == "res":
            print("GraphConv->LN/BN->ReLU->Res")
        elif self.block == "dense":
            raise NotImplementedError("To be implemented")
        elif self.block == "plain":
            print("GraphConv->LN/BN->ReLU")
        else:
            raise Exception("Unknown block Type")

        if self.add_virtual_node:
            self.virtualnode_embedding = torch.nn.Embedding(1, hidden_channels)
            torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

            self.mlp_virtualnode_list = torch.nn.ModuleList()

            for layer in range(num_gc_layers - 1):
                self.mlp_virtualnode_list.append(MLP([hidden_channels] * 3, norm=norm))

        for _ in range(num_gc_layers):
            conv = GENConv(
                hidden_channels,
                hidden_channels,
                aggr=aggr,
                t=t,
                learn_t=self.learn_t,
                p=p,
                learn_p=self.learn_p,
                y=y,
                learn_y=self.learn_p,
                msg_norm=self.msg_norm,
                learn_msg_scale=learn_msg_scale,
                encode_edge=self.conv_encode_edge,
                bond_encoder=True,
                norm=norm,
                mlp_layers=mlp_layers,
            )
            self.gcns.append(conv)
            self.norms.append(norm_layer(norm, hidden_channels))

        self.atom_encoder = AtomEncoder(emb_dim=hidden_channels)

        if not self.conv_encode_edge:
            self.bond_encoder = BondEncoder(emb_dim=hidden_channels)

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise Exception("Unknown Pool Type")

        self.set2set = Set2Set(hidden_channels, processing_steps=3)

    def forward(self, x, edge_index, edge_attr, batch, classification=True):
        h = self.atom_encoder(x)

        if self.add_virtual_node:
            virtualnode_embedding = self.virtualnode_embedding(
                torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device)
            )
            h = h + virtualnode_embedding[batch]

        if self.conv_encode_edge:
            edge_emb = edge_attr
        else:
            edge_emb = self.bond_encoder(edge_attr)

        if self.block == "res+":
            xs = []
            h = self.gcns[0](h, edge_index, edge_emb)
            for layer in range(1, self.num_gc_layers):
                h1 = self.norms[layer - 1](h)
                h2 = F.relu(h1)
                h2 = F.dropout(h2, p=self.dropout, training=self.training)
                xs.append(h2)
                if self.add_virtual_node:
                    virtualnode_embedding_temp = global_add_pool(h2, batch) + virtualnode_embedding
                    virtualnode_embedding = F.dropout(
                        self.mlp_virtualnode_list[layer - 1](virtualnode_embedding_temp),
                        self.dropout,
                        training=self.training,
                    )
                    h2 = h2 + virtualnode_embedding[batch]

                h = self.gcns[layer](h2, edge_index, edge_emb) + h

            h = self.norms[self.num_gc_layers - 1](h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            xs.append(h)
        elif self.block == "res":

            h = F.relu(self.norms[0](self.gcns[0](h, edge_index, edge_emb)))
            h = F.dropout(h, p=self.dropout, training=self.training)

            for layer in range(1, self.num_gc_layers):
                h1 = self.gcns[layer](h, edge_index, edge_emb)
                h2 = self.norms[layer](h1)
                h = F.relu(h2) + h
                h = F.dropout(h, p=self.dropout, training=self.training)

        elif self.block == "plain":

            h = F.relu(self.norms[0](self.gcns[0](h, edge_index, edge_emb)))
            h = F.dropout(h, p=self.dropout, training=self.training)

            for layer in range(1, self.num_gc_layers):
                h1 = self.gcns[layer](h, edge_index, edge_emb)
                h2 = self.norms[layer](h1)
                if layer != (self.num_gc_layers - 1):
                    h = F.relu(h2)
                else:
                    h = h2
                h = F.dropout(h, p=self.dropout, training=self.training)
        else:
            raise Exception("Unknown block Type")

        if classification:
            h_graph = self.set2set(h, batch)
            xout = h_graph
        else:
            h_graph = self.pool(h, batch)
            xout = h_graph

        return xout

    def get_embeddings(self, loader):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                data.to(device)
                x, edge_index, edge_attr, batch = (
                    data.x,
                    data.edge_index,
                    data.edge_attr,
                    data.batch,
                )
                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(device)
                x, _ = self.forward(x, edge_index, edge_attr, batch)
                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y


class SupConDeeperGCN(torch.nn.Module):
    def __init__(self, opt: Any):
        """The molecular graph branch.

        Args:
            opt (Any): Parsed arguments.
        """
        super(SupConDeeperGCN, self).__init__()
        dim = 256
        num_gc_layers = opt.num_gc_layers
        dropout = 0.2
        block = "res+"
        conv_encode_edge = False
        add_virtual_node = True
        hidden_channels = 256
        self.num_tasks = opt.num_tasks
        aggr = "max"
        graph_pooling = "sum"
        learn_t = False
        t = 0.1
        mlp_layers = opt.mlp_layers
        self.classification = opt.classification
        if opt.classification:
            norm = "batch"
        else:
            norm = "layer"

        self.encoder = DeeperGCN(
            num_gc_layers,
            dropout,
            block,
            conv_encode_edge,
            add_virtual_node,
            hidden_channels,
            self.num_tasks,
            aggr,
            graph_pooling,
            t=t,
            learn_t=learn_t,
            mlp_layers=mlp_layers,
            norm=norm,
        )
        if opt.classification:
            self.dense = torch.nn.Linear(dim * 2, 128)
        else:
            self.dense = torch.nn.Linear(dim, 128)

        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, batch: Tensor):
        """Generate the embedding of the molecular graph branch.
        Args:
            batch (Tensor) : A batch of data of the current epoch.

        Returns:
            The embedding of the molecular graph branch.
        """
        feat = self.encoder(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch, self.classification
        )
        feat = self.dropout(feat)
        feat = self.dense(feat)
        return feat
