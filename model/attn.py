import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy
import numpy as np

import random
import math
import time

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')


def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


SRC = Field(tokenize=tokenize_de,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
            batch_first=True)

TRG = Field(tokenize=tokenize_en,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
            batch_first=True)

train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                    fields=(SRC, TRG))

SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device)


class MHALayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        assert hid_dim % n_heads == 0, 'hid_dim should be divided'
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

        self.fc_Q = nn.Linear(hid_dim, hid_dim)
        self.fc_K = nn.Linear(hid_dim, hid_dim)
        self.fc_V = nn.Linear(hid_dim, hid_dim)
        self.fc_O = nn.Linear(hid_dim, hid_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        Q = self.fc_Q(query).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.fc_K(key).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.fc_V(value).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        E = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            E = E.masked_fill(mask == 0, -1e10)
        attn = torch.matmul(self.dropout(torch.softmax(E, dim=-1)), V)
        attn = attn.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.hid_dim)
        return self.fc_O(attn)
