"""
Spatio-Temporal GCN + LSTM Model for Traffic Congestion Prediction.
Combines Graph Convolutional Networks (spatial) with LSTM (temporal).
"""
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
 
 
class GraphConvolution(nn.Module):
    """Single Graph Convolutional Layer using adjacency matrix."""
 
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self._reset_parameters()
 
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
 
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : (batch, nodes, in_features)
            adj : (nodes, nodes) normalized adjacency matrix
        Returns:
            out : (batch, nodes, out_features)
        """
        support = torch.matmul(x, self.weight)      # (B, N, out)
        out = torch.matmul(adj, support)            # (B, N, out) -- broadcast
        if self.bias is not None:
            out = out + self.bias
        return F.relu(out)
 
 
class STGCNBlock(nn.Module):
    """One Spatio-Temporal block: GCN -> BatchNorm -> Dropout."""
 
    def __init__(self, in_features: int, hidden: int, out_features: int, dropout: float = 0.3):
        super(STGCNBlock, self).__init__()
        self.gc1 = GraphConvolution(in_features, hidden)
        self.gc2 = GraphConvolution(hidden, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.gc1(x, adj)
        x = self.gc2(x, adj)
        # BN expects (B*N, n_feat); reshape accordingly
        B, N, n_feat = x.shape                          # renamed F -> n_feat
        x = self.bn(x.view(B * N, n_feat)).view(B, N, n_feat)
        return self.dropout(x)
 
 
class TrafficSTGCN_LSTM(nn.Module):
    """
    Full Spatio-Temporal model:
      Input  -> STGCNBlock -> LSTM (temporal) -> FC -> congestion score per node
 
    Input shape : (batch, seq_len, nodes, node_features)
    Output shape: (batch, nodes) -- predicted congestion score [0, 1]
    """
 
    def __init__(
        self,
        node_features: int,
        gcn_hidden: int = 64,
        gcn_out: int = 32,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        num_nodes: int = 50,
        dropout: float = 0.3,
    ):
        super(TrafficSTGCN_LSTM, self).__init__()
        self.num_nodes = num_nodes
        self.gcn_out = gcn_out
 
        self.stgcn = STGCNBlock(node_features, gcn_hidden, gcn_out, dropout)
 
        # After GCN, flatten node features for LSTM input: (B, seq_len, N*gcn_out)
        self.lstm = nn.LSTM(
            input_size=num_nodes * gcn_out,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_hidden, 64)
        self.fc2 = nn.Linear(64, num_nodes)     # one score per node
 
    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        hidden=None,
    ):
        """
        Args:
            x   : (B, T, N, n_feat)
            adj : (N, N)
        Returns:
            out : (B, N) -- congestion scores per node for next time step
        """
        B, T, N, n_feat = x.shape               # renamed F -> n_feat
        gcn_out_seq = []
        for t in range(T):
            xt = x[:, t, :, :]                  # (B, N, n_feat)
            zt = self.stgcn(xt, adj)            # (B, N, gcn_out)
            gcn_out_seq.append(zt.view(B, -1))  # (B, N*gcn_out)
 
        lstm_in = torch.stack(gcn_out_seq, dim=1)   # (B, T, N*gcn_out)
        lstm_out, hidden = self.lstm(lstm_in, hidden)
        last = lstm_out[:, -1, :]                    # (B, lstm_hidden)
 
        out = F.relu(self.fc1(self.dropout(last)))   # (B, 64)
        out = torch.sigmoid(self.fc2(out))           # (B, N) -> [0,1]
        return out, hidden
 