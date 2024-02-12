"""
Author: Chunhui Gu
Email: fduguchunhui@gmail.com
Created: 2/11/24
"""
import torch

class ResidualAdd(torch.nn.Module):
    def forward(self, x, x_residual):
        return x + x_residual
