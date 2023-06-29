import torch
from torch.nn import Conv1d, Embedding, Linear, MaxPool1d, ModuleList
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, SAGEConv,SortPooling

class DGCNN(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        num_layers,
        max_z,
        k,
        feature_dim=0,
        GNN=GraphConv,
        dropout=0.0,
    ):
        super(DGCNN, self).__init__()

        self.feature_dim = feature_dim
        self.dropout = dropout

        self.k = k
        self.sort_pool = SortPooling(k=self.k)

        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)

        self.convs = ModuleList()
        initial_channels = hidden_channels + self.feature_dim

        self.convs.append(GNN(initial_channels, hidden_channels))

        for _ in range(0, num_layers - 1):
            self.convs.append(GNN(hidden_channels, hidden_channels))

        self.convs.append(GNN(hidden_channels, 1))

        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(
            conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1
        )
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(dense_dim, 128)
        self.lin2 = Linear(128, 1)

    def forward(self, g, z, x=None, edge_weight=None):
        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if x is not None:
            x = torch.cat([z_emb, x.to(torch.float)], 1)
        else:
            x = z_emb
        xs = [x]

        for conv in self.convs:
            xs += [
                F.dropout(
                    torch.tanh(conv(g, xs[-1], edge_weight=edge_weight)),
                    p=self.dropout,
                    training=self.training,
                )
            ]
        x = torch.cat(xs[1:], dim=-1)

        # global pooling
        x = self.sort_pool(g, x)
        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        x = F.relu(self.conv1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # [num_graphs, dense_dim]

        # MLP.
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x


class GCN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        dropout,
        dataset,
    ):
        super(GCN, self).__init__()

        self.dataset = dataset
        self.convs = torch.nn.ModuleList()

        self.convs.append(GraphConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GraphConv(hidden_channels, hidden_channels)
            )
        self.convs.append(GraphConv(hidden_channels, out_channels))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, g, x):
        for conv in self.convs[:-1]:
            x = conv(g, x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](g, x)
        return x


class SAGE(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        dropout,
        dataset,
        reduce="mean",
    ):
        super(SAGE, self).__init__()

        self.dataset = dataset
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, reduce))
        for _ in range(num_layers - 2):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels, reduce))
        self.convs.append(SAGEConv(hidden_channels, out_channels, reduce))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, g, x):
        for conv in self.convs[:-1]:
            x = conv(g, x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](g, x)
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, num_layers, dropout
    ):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.lins.append(Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)
