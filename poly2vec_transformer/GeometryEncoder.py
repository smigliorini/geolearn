# source: https://github.com/jlliRUC/Poly2Vec_GeoAI/blob/main/models/GeometryEncoder.py
import torch
import torch.nn as nn
from poly2vec import Poly2Vec

class GeometryEncoder(nn.Module):
    """
    GeometryEncoder: A wrapper of the whole framework for the spatial reasoning task.
    """

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.encoder = Poly2Vec(device=self.device)
    def forward(self, x, lengths, dataset_type):
        x_emb = self.encoder.encode(x, lengths, dataset_type=dataset_type)

        return x_emb