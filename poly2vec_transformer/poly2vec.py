# source: https://github.com/jlliRUC/Poly2Vec_GeoAI/blob/main/models/poly2vec.py

import torch
import torch.nn as nn
from fourier_encoder import GeometryFourierEncoder


class MLP(nn.Module):
    # For fusion weights
    def __init__(self, d_model, dropout=0.1):
        super(MLP, self).__init__()
        self.w_1 = nn.Linear(d_model, d_model * 2)
        self.w_2 = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class SimpleMLP(nn.Module):
    def __init__(self, d_input, d_hid, d_out, dropout):
        super(SimpleMLP, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(d_input, d_hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hid, d_out)
            # nn.LayerNorm(d_out, eps=1e-6)
        )

    def forward(self, x):
        return self.nn(x)


class EnhancedMLP(nn.Module):
    def __init__(self, d_input, d_hid, d_out, dropout, layers=4):
        super(EnhancedMLP, self).__init__()
        self.layers = nn.ModuleList([nn.Sequential(
            nn.Linear(d_input if i == 0 else d_hid, d_hid),
            nn.BatchNorm1d(d_hid),
            nn.ReLU(),
            nn.Dropout(dropout)
        ) for i in range(layers)])
        self.output_layer = nn.Linear(d_hid, d_out)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)


class Poly2Vec(nn.Module):
    """
    Poly2Vec model: Learns an embedding vector for any polymorphic 2D shape.
        Operates on encoded Fourier features of the shapes.
    """

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.fusion = 'learned_fusion'
        self.d_input = 210
        self.d_hid, self.d_out, self.dropout = 100, 32, 0.2
        # simple MLP
        if self.fusion == 'learned_fusion' or self.fusion == 'concat':
            d_input = 2 * self.d_input
        else:
            d_input = self.d_input
        self.nn = SimpleMLP(d_input, self.d_hid, self.d_out, self.dropout)
        # fourier encoder
        self.ft_encoder = GeometryFourierEncoder(device=self.device)
        self.param_mag = nn.Sequential(nn.Linear(self.d_input, self.d_input, bias=False),
                                       MLP(self.d_input, dropout=self.dropout))
        self.param_phase = nn.Sequential(nn.Linear(self.d_input, self.d_input, bias=False),
                                         MLP(self.d_input, dropout=self.dropout))

    def forward(self, x, lengths, dataset_type):
        B = x.shape[0]
        x_encoded = self.ft_encoder.encode(x, lengths, dataset_type=dataset_type)
        x_phase = torch.angle(x_encoded).reshape(B, -1)
        x_mag = torch.abs(x_encoded).reshape(B, -1)
        if self.fusion == "learned_fusion":
            x_mag, x_phase = self.param_mag(x_mag), self.param_phase(x_phase)
            x_ = torch.concat([x_mag, x_phase], dim=1).reshape(B, -1)
        elif self.fusion == "mag":
            x_ = x_mag
        elif self.fusion == "phase":
            x_ = x_phase
        elif self.fusion == "concat":
            x_  = torch.cat([x_mag, x_phase], dim=1).reshape(B, -1)
        
        x_emb = self.nn(x_)

        return x_emb

    def encode(self, x, lengths, dataset_type):

        return self.forward(x, lengths, dataset_type)