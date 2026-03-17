import torch
import torch
from torch_geometric.nn import MLP, fps, global_max_pool, radius
from torch_geometric.nn import PointNetConv

import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

class SetAbstraction(torch.nn.Module):
    def __init__(self, ratio, radius, max_neighbors, mlp):
        super().__init__()
        self.ratio = ratio
        self.r = radius
        self.max_neighbors = max_neighbors
        self.conv = PointNetConv(MLP(mlp), add_self_loops=True)

    def forward(self, x, pos, batch):
        idx = list(range(pos.shape[0])) if self.ratio == 1.0 else fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                        max_num_neighbors=self.max_neighbors)
        edge_index = torch.stack([col, row], dim=0)
        
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch

class GlobalSetAbstraction(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), pos.shape[0]))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class PointNet(torch.nn.Module):
    def __init__(
        self,
        set_abstractions=[
            {'ratio': 0.75, 'radius': 0.5, 'max_neighbors': 64, 'mlp': [3, 64, 64, 128]},
            {'ratio': 0.75, 'radius': 0.5, 'max_neighbors': 64, 'mlp': [3, 64, 64, 128]}
        ],
        global_abstraction={'mlp': [256 + 3, 256, 512, 1024]},
        final_mlp=[1024, 512, 256, 9],
        dropout=0.1
    ):
        super().__init__()

        # Input channels account for both `pos` and node features.
        self.sa1 = SetAbstraction(**set_abstractions[0])
        self.sa2 = SetAbstraction(**set_abstractions[1])
        self.global_abstraction = GlobalSetAbstraction(MLP(global_abstraction['mlp']))
        self.mlp = MLP(final_mlp, dropout=dropout, norm=None)

    def forward(self, data):
        sa_0 = (data.x, data.pos, data.batch)
        sa_1 = self.sa1(*sa_0)
        sa_2 = self.sa2(*sa_1)
        ga_out = self.global_abstraction(*sa_2)
        x, pos, batch = ga_out
        return self.mlp(x)