import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from models.deepgcn_vertex import GENConv
from models.deepgcn_nn import AtomEncoder, BondEncoder, MLP, norm_layer


class DeeperGCN(torch.nn.Module):
    def __init__(
        self,
        dim,
        num_gc_layers,
        dropout,
        block,
        conv_encode_edge,
        add_virtual_node,
        hidden_channels,
        num_tasks,
        aggr="add",
        t=1.0,
        learn_t=False,
        p=1.0,
        learn_p=False,
        y=0.0,
        learn_y=False,
        mlp_layers=1,
        norm="batch",
    ):
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

        norm = "batch"
        mlp_layers = mlp_layers
        graph_pooling = "sum"

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
            # self.norms.append(torch.nn.BatchNorm1d(hidden_channels, affine=True))

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

    def forward(self, x, edge_index, edge_attr, batch):
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

        h_graph = self.pool(h, batch)
        xout = h_graph

        return xout

    def get_embeddings(self, loader):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = 'cpu'
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
    """backbone + projection head"""

    def __init__(self, num_tasks:int=1, mlp_layers:int=1, num_gc_layers:int=7):
        """The molecular graph branch. 

        Args:
            num_tasks (int, optional): Number of tasks in the dataset. Defaults to 1.
            mlp_layers (int, optional): Number of mlp layers. Defaults to 1.
            num_gc_layers (int, optional): Depth of the deepergcn model. Defaults to 7.
        """
        super(SupConDeeperGCN, self).__init__()
        dim = 256
        num_gc_layers = num_gc_layers
        dropout = 0.2
        block = "res+"
        conv_encode_edge = False
        add_virtual_node = True
        hidden_channels = 256
        self.num_tasks = num_tasks
        aggr = "max"
        learn_t = False
        t = 0.1
        mlp_layers = mlp_layers
        self.encoder = DeeperGCN(
            dim,
            num_gc_layers,
            dropout,
            block,
            conv_encode_edge,
            add_virtual_node,
            hidden_channels,
            self.num_tasks,
            aggr,
            t=t,
            learn_t=learn_t,
            mlp_layers=mlp_layers,
            norm="layer",
        )

        self.dense = torch.nn.Linear(256, 128)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, batch:Tensor):
        """
        Args:
            batch (Tensor)

        Returns:
            The embedding of the molecular graph branch. 
        """
        feat = self.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        feat = self.dropout(feat)
        feat = self.dense(feat)
        return feat
