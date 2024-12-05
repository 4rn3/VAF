import math

import numpy as np

import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange, reduce

### Super Simple test network

class Unconditioned_Simple_MLP(nn.Module):
    def __init__(self, io_dim, hidden_size):
        super(Unconditioned_Simple_MLP, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(io_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, io_dim)
        )

    def forward(self, x):
        return self.model(x)

####################################################################
### Simple MLP network
 
class Simple_FF(nn.Module):
    def __init__(self, hidden_size):
        """
        hidden_size: list of layer dimensions, e.g., [1024, 512, 256]
        """
        super(Simple_FF, self).__init__()
        
        self.layers = self.init_layers(hidden_size)
        self.activation = nn.ReLU(inplace=True)
    
    def init_layers(self, layer_dim):
        """
        Initialize layers based on dimensions in `layer_dim`.
        """
        layers = nn.ModuleList()
        
        for i in range(len(layer_dim) - 1):
            layers.append(nn.Linear(layer_dim[i], layer_dim[i+1], bias=True))
        
        return layers
            
    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return x
    
class Unconditioned_MLP(nn.Module):
    def __init__(self, io_dim, hidden_size):
        """
        hidden_size: list of layer dimensions, e.g., [1024, 512, 256]
        """
        super(Unconditioned_MLP, self).__init__()
        
        self.input = nn.Linear(io_dim, hidden_size[0])
        self.activation = nn.ReLU(inplace=True)
        self.feed_forward = Simple_FF(hidden_size)
        self.output = nn.Linear(hidden_size[-1], io_dim)
        
    def forward(self, x):
        x = self.activation(self.input(x))
        x = self.feed_forward(x)
        x = self.activation(self.output(x))
        
        return x

####################################################################
### Simple MLP with time embedding network
### This is basically the FinDiff implementation but for a univariate time series without categorical embeddings: https://github.com/sattarov/FinDiff/tree/main

class Unconditioned_TimeEmbedding_MLP(nn.Module):
    def __init__(self, io_dim, hidden_size, time_dim=64):
        """
        hidden_size: list of layer dimensions, e.g., [1024, 1024, 1024]
        """
        super(Unconditioned_TimeEmbedding_MLP, self).__init__()
        
        self.io_dim = io_dim
        self.time_dim = time_dim
        
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim)
        )
        
        self.proj = nn.Sequential(
            nn.Linear(self.io_dim, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim)
        )
        
        self.activation = nn.ReLU(inplace=True)
        self.feed_forward = Simple_FF([self.time_dim, *hidden_size])
        self.output = nn.Linear(hidden_size[-1], io_dim)
        
        
    def timestep_embedding(self, timesteps, dim_out, max_period=10000):
        half = dim_out // 2
        
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=timesteps.device)

        args = timesteps[:, None].float() * freqs[None]
        
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if dim_out % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        
        # print(f"Output shape for timestep_embedding fn: {embedding.shape}")
        
        return embedding

        
    def forward(self, x, timesteps):
        # print(f"froward fn X shape: {x.shape}")
        # print(f"froward fn timestep shape: {timesteps.shape}")
        emb = self.time_embed(self.timestep_embedding(timesteps, self.time_dim))
        # print(f"embedded shape forward:{emb.shape}")
        
        x = self.proj(x) + emb
        
        x = self.feed_forward(x)
        x = self.activation(self.output(x))
        
        return x

####################################################################
### Attempt at implementing a uncondioned transform network
### basically this implementation: https://github.com/fahim-sikder/TransFusion
    
class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)
    
    
class TransEncoder(nn.Module):
    
    def __init__(self, features=1, latent_dim=256, num_heads=4, num_layers=6, dropout=0.1, activation='gelu', ff_size=1024):
        
        super().__init__()

        self.channels = features
        self.latent_dim  = latent_dim
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.activation = activation
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.pos_enc = PositionalEncoding(self.latent_dim)
        self.emb_timestep = TimestepEmbedder(self.latent_dim, self.pos_enc)

        self.input_dim = nn.Linear(self.channels, self.latent_dim) #check how original work gets to these dim
        self.output_dim = nn.Linear(self.latent_dim, self.channels)
        
        self.TransEncLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                  nhead=self.num_heads,
                                                  dim_feedforward=self.ff_size,
                                                  dropout=self.dropout,
                                                  activation=self.activation)

        self.TransEncodeR = nn.TransformerEncoder(self.TransEncLayer,
                                                     num_layers=self.num_layers)
        
        
    def forward(self, x, t):
        print(f"shape of x: {x.shape}")
        x = torch.transpose(x, 1, 2)
        print(f"shape of x transposed: {x.shape}") 
        
        x = self.input_dim(x) #goes wrong here
        x = torch.transpose(x, 0, 1)
        print(f"shape of x transposed 2: {x.shape}") 
        
        embed = self.emb_timestep(t)
        time_added_data = torch.cat((embed, x), axis = 0)
        time_added_data = self.pos_enc(time_added_data)
        
        trans_output = self.TransEncodeR(time_added_data)[1:] #Default:(seq, batch, feature), batch first: True (batch, seq, feature)
        final_output = self.output_dim(trans_output)
        
        transposed_data = final_output.permute(1, 2, 0)
        print(f"shape of transposed_data: {transposed_data.shape}")
        return transposed_data