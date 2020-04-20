import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from tokenizer import Tokenizer

VOCAB_PATH = "../vocab.txt"

class Model(nn.Module):

    KEYS = ['location', 'missing_items', 'observation', 'missing_utilities', 'current_inventory']

    def __init__(self, device, hidden_size=64, bidirectional=True, hidden_linear_size=128):

        super(Model, self).__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.obs_encoded_hidden_size = self.hidden_size * len(self._KEYS) * (2 if bidirectional else 1)
        self.cmd_encoded_hidden_size = self.hidden_size * (2 if bidirectional else 1)
        self.state_hidden = None

        self.tokenizer = Tokenizer(device=device)

        # Encoder for the commands
        # self.cmd_encoder = nn.GRU(self.embedding_dim, self.hidden_size, batch_first=True, bidirectional=bidirectional)

        # RNN that keeps track of the encoded state over time
        self.state_gru = nn.GRU(self.obs_encoded_hidden_size, self.obs_encoded_hidden_size, batch_first=True)

        # Critic to determine a value for the current state
        self.critic = nn.Sequential(nn.Linear(self.obs_encoded_hidden_size, hidden_linear_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_linear_size, 1))

        # # Scorer for the commands
        self.att_cmd = nn.Sequential(nn.Linear(self.obs_encoded_hidden_size + self.cmd_encoded_hidden_size, hidden_linear_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_linear_size, 1))

        self.to(self.device)

