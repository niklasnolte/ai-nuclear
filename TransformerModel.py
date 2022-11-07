from data import get_data
import torch.nn as nn
import torch
import torch.nn.functional as F
import math,copy,re
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import torchtext
import matplotlib.pyplot as plt
from train_model import train
import random


warnings.simplefilter("ignore")


DROPOUT_VALUE = 0.2
random.seed(42)
#https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch/notebook
class Embedding(nn.Module):
    def __init__(self, n_protons, n_neutrons,  embed_dim):
        """
        Args:
            n_protons, n_neutrons: number of neturons/protons
            embed_dim: dimension of embeddings
        """
        super(Embedding, self).__init__()
        self.emb_proton = nn.Embedding(n_protons, embed_dim) # [ batch_size, hidden_dim ]
        self.emb_neutron = nn.Embedding(n_neutrons, embed_dim) # [ batch_size, hidden_dim ]
        self.emb_proton.weight.data.uniform_(-1,1)
        self.emb_neutron.weight.data.uniform_(-1,1)
    
    def forward(self, x):
        """
        Args:
            x: input vector (protons, neutrons stacked emmbeddings)
        Returns:
            x: embedding vector
        """
        proton = self.emb_proton(x[:,0]) # [ batch_size, hidden_dim ]
        neutron = self.emb_neutron(x[:,1]) # [ batch_size, hidden_dim ]
        out = torch.stack((proton, neutron), dim = 1)   
        '''
        seems unnecessary
        # make embeddings relatively larger
        x = x * math.sqrt(self.embed_dim)
        '''
        return out



class PositionalEmbedding(nn.Module):
    def __init__(self,max_seq_len,embed_dim):
        """
        Args:
            seq_len: length of input sequence
            embed_model_dim: demension of embedding
        """
        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_dim

        pe = torch.zeros(max_seq_len,self.embed_dim)
        for pos in range(max_seq_len):
            for i in range(0,self.embed_dim,2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/self.embed_dim)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)


    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            x: output
        """
      
        # make embeddings relatively larger
        x = x * math.sqrt(self.embed_dim)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:,:seq_len], requires_grad=False)
        return x
               


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=512, n_heads=8):
        """
        Args:
            embed_dim: dimension of embeding vector output
            n_heads: number of self attention heads
        """
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim   #512 dim
        self.n_heads = n_heads   #8
        self.single_head_dim = int(self.embed_dim / self.n_heads)   #512/8 = 64  . each key,query, value will be of 64d
       
        #key,query and value matrixes    #64 x 64   
        self.query_matrix = nn.Linear(self.single_head_dim , self.single_head_dim ,bias=False)  # single key matrix for all 8 keys #512x512
        self.key_matrix = nn.Linear(self.single_head_dim  , self.single_head_dim, bias=False)
        self.value_matrix = nn.Linear(self.single_head_dim ,self.single_head_dim , bias=False)
        self.out = nn.Linear(self.n_heads*self.single_head_dim ,self.embed_dim) 

    def forward(self,key,query,value,mask=None):    #batch_size x sequence_length x embedding_dim    # 32 x 2 x 512

        """
        Args:
           key : key vector
           query : query vector
           value : value vector
           mask: mask for decoder
        
        Returns:
           output vector from multihead attention
        """
        batch_size = key.size(0)
        seq_length = key.size(1)
        
        # query dimension can change in decoder during inference. 
        # so we cant take general seq_length
        seq_length_query = query.size(1)
        
        #we want to learn the key, query value matrices. They all operate on the embedding
        # 32x2x512
        key = key.view(batch_size, seq_length, self.n_heads, self.single_head_dim)  #batch_size x sequence_length x n_heads x single_head_dim = (32x2x8x64)
        query = query.view(batch_size, seq_length_query, self.n_heads, self.single_head_dim) #(32x2x8x64)
        value = value.view(batch_size, seq_length, self.n_heads, self.single_head_dim) #(32x2x8x64)
       
        k = self.key_matrix(key)       # (32x2x8x64)
        q = self.query_matrix(query)   
        v = self.value_matrix(value)

        q = q.transpose(1,2)  # (batch_size, n_heads, seq_len, single_head_dim)    # (32 x 8 x 2 x 64)
        k = k.transpose(1,2)  # (batch_size, n_heads, seq_len, single_head_dim)
        v = v.transpose(1,2)  # (batch_size, n_heads, seq_len, single_head_dim)
       
        # computes attention
        # adjust key for matrix multiplication
        k_adjusted = k.transpose(-1,-2)  #(batch_size, n_heads, single_head_dim, seq_ken)  #(32 x 8 x 64 x 2)
        product = torch.matmul(q, k_adjusted)  #(32 x 8 x 2 x 64) x (32 x 8 x 64 x 2) = #(32x8x2x2)
      
        
        # fill those positions of product matrix as (-1e20) where mask positions are 0
        if mask is not None:
             product = product.masked_fill(mask == 0, float("-1e20"))

        #divising by square root of key dimension
        product = product / math.sqrt(self.single_head_dim) # / sqrt(64)

        #applying softmax
        scores = F.softmax(product, dim=-1)
 
        #mutiply with value matrix
        scores = torch.matmul(scores, v)  ##(32x8x 2x 2) x (32 x 8 x 2 x 64) = (32 x 8 x 2 x 64) 
        
        #concatenated output
        concat = scores.transpose(1,2).contiguous().view(batch_size, seq_length_query, self.single_head_dim*self.n_heads)  # (32x8x2x64) -> (32x2x8x64)  -> (32,2,512)
        
        output = self.out(concat) #(32,2,512) -> (32,2,512)
       
        return output


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
        super(TransformerBlock, self).__init__()
        
        """
        Args:
           embed_dim: dimension of the embedding
           expansion_factor: fator ehich determines output dimension of linear layer
           n_heads: number of attention heads
        
        """
        self.attention = MultiHeadAttention(embed_dim, n_heads)
        
        self.norm1 = nn.LayerNorm(embed_dim) 
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
                          nn.Linear(embed_dim, expansion_factor*embed_dim),
                          nn.ReLU(),
                          nn.Linear(expansion_factor*embed_dim, embed_dim)
        )

        self.dropout1 = nn.Dropout(DROPOUT_VALUE)
        self.dropout2 = nn.Dropout(DROPOUT_VALUE)

    def forward(self,key,query,value):
        
        """
        Args:
           key: key vector
           query: query vector
           value: value vector
           norm2_out: output of transformer block
        
        """
        
        attention_out = self.attention(key,query,value)  #32x2x512
        attention_residual_out = attention_out + value  #32x2x512
        norm1_out = self.dropout1(self.norm1(attention_residual_out)) #32x2x512

        feed_fwd_out = self.feed_forward(norm1_out) #32x2x512 -> #32x2x2048 -> 32x2x512
        feed_fwd_residual_out = feed_fwd_out + norm1_out #32x2x512
        norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out)) #32x2x512

        return norm2_out

class TransformerEncoder(nn.Module):
    """
    Args:
        seq_len : length of input sequence
        embed_dim: dimension of embedding
        num_layers: number of encoder layers
        expansion_factor: factor which determines number of linear layers in feed forward layer
        n_heads: number of heads in multihead attention
        
    Returns:
        out: output of the encoder
    """
    def __init__(self, n_protons, n_neutrons, embed_dim, seq_len = 2, num_layers=2, expansion_factor=4, n_heads=8):
        super(TransformerEncoder, self).__init__()
        
        self.embedding_layer = Embedding(n_protons, n_neutrons, embed_dim)
        self.positional_encoder = PositionalEmbedding(seq_len, embed_dim)

        self.layers = nn.ModuleList([TransformerBlock(embed_dim, expansion_factor, n_heads) for i in range(num_layers)])
        #how many encoder alyers do we want 

    def forward(self, x):
        embed_out = self.embedding_layer(x)
        out = self.positional_encoder(embed_out)
        for layer in self.layers:
            out = layer(out,out,out)
        return out  #32x2x512

class TransformerModel(nn.Module):
    def __init__(self, n_protons, n_neutrons, embed_dim, num_layers=6, expansion_factor=4, n_heads=8):
        super(TransformerModel, self).__init__()
        
        """  
        Args:
           embed_dim:  dimension of embedding 
           seq_length : length of input sequence
           num_layers: number of encoder layers
           expansion_factor: factor which determines number of linear layers in feed forward layer
           n_heads: number of heads in multihead attention
        
        """
        

        self.encoder = TransformerEncoder(n_protons=n_protons, 
                                        n_neutrons = n_neutrons, 
                                        embed_dim = embed_dim, 
                                        num_layers = num_layers,
                                        expansion_factor = expansion_factor,
                                        n_heads = n_heads)
        self.emb_proton = self.encoder.embedding_layer.emb_proton
        self.emb_neutron = self.encoder.embedding_layer.emb_neutron

        self.nonlinear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2*embed_dim, embed_dim), # we need 2*hidden_dim to get proton and neutron embedding
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1))

    def forward(self, x): # x: [ batch_size, 2 [n_protons, n_neutrons] ]
        encoded = self.encoder(x) # batch size x 2 x embedding_dim
        flattened = torch.cat((encoded[:,0, :], encoded[:,1,:]), dim=1) # batch size x (2*embedding_dim)
        x = self.nonlinear(flattened)
        return x

if __name__ == '__main__':
    '''
    embed_dim = 32
    X_train, _, _, _, (n_protons, n_neutrons) = get_data() 
    emb_var = Embedding(n_protons, n_neutrons, embed_dim)
    emb = emb_var.forward(X_train)
    pos_emb_var = PositionalEmbedding(max_seq_len=2, embed_dim = embed_dim)
    pos_emb = pos_emb_var.forward(emb)
    '''

    
    
    train(modelclass=TransformerModel, 
        lr=(2e-3)/4, 
        wd=1e-4, 
        embed_dim=64, 
        basepath="models/Transformer_posemb/", 
        device=torch.device("cuda"),
        title = 'Transformer_posemb'
        )

    train(modelclass=TransformerModel, 
        lr=(2e-3)/4, 
        wd=1e-4, 
        embed_dim=64, 
        basepath="models/Transformer_posemb_small/", 
        device=torch.device("cuda"),
        title = 'Transformer_posemb_small',
        num_layers = 1, 
        expansion_factor = 1, 
        n_heads = 1
        )
    
    '''

    train(modelclass=TransformerModel, 
        lr=(2e-3), 
        wd=1e-4, 
        embed_dim=512, 
        basepath="models/TransformerBase/", 
        device=torch.device("cuda"),
        title = 'TransformerBase'
        )
    train(modelclass=TransformerModel, 
        lr=(2e-3), 
        wd=1e-4, 
        embed_dim=64, 
        basepath="models/Transformer_1layer/", 
        device=torch.device("cuda"),
        title = 'Transformer_1layer',
        num_layers=1
        )
    train(modelclass=TransformerModel, 
        lr=2e-3, 
        wd=1e-4, 
        embed_dim=64, 
        basepath="models/TransformerModel2exp/", 
        device=torch.device("cuda"),
        title = 'Transformer_2exp',
        expansion_factor = 2
        )
    '''

