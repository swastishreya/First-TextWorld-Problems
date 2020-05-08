import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# from bert_tokenizer import Tokenizer
from sentence_tokenizer import Tokenizer # using sentence BERT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):

    STATE_LIST = ['location', 'missing_items', 'observation', 'missing_utilities', 'current_inventory']

    def __init__(self, device, hidden_size=64, bidirectional=True, hidden_linear_size=128):

        super(Model, self).__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.obs_encoded_hidden_size = self.hidden_size * len(self.STATE_LIST) * (2 if bidirectional else 1)
        self.cmd_encoded_hidden_size = self.hidden_size * (2 if bidirectional else 1)
        self.state_hidden = None
        # TO DO - add graph embedding_dim

        self.tokenizer = Tokenizer(device=device)
        self.embedding_dim = self.tokenizer.embedding_dim

        # # RNN that keeps track of the encoded state over time
        self.state_gru = nn.GRU(self.obs_encoded_hidden_size, self.obs_encoded_hidden_size, batch_first=True)

        # # Critic to determine a value for the current state
        # TO DO - add graph embedding_dim
        self.critic = nn.Sequential(nn.Linear(self.obs_encoded_hidden_size, hidden_linear_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_linear_size, 1))

        # # # Scorer for the commands
        # TO DO - add graph embedding_dim
        self.att_cmd = nn.Sequential(nn.Linear(self.obs_encoded_hidden_size + self.cmd_encoded_hidden_size, hidden_linear_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_linear_size, 1))

        self.to(self.device)

    def forward(self, state_description, commands):
        """
        :param state_description: Dictionary of strings with keys=STATE_LIST that represents the current game state
        :param commands: Set of possible commands
        :return: Best command from set of possible commands
        """
        command_strings = commands

        # Encode the state_description
        obs_encoded = self.tokenizer.encode_state(state_description)

        if self.state_hidden is None:
            self.state_hidden = torch.zeros((1, 1, self.obs_encoded_hidden_size), device=self.device)

        # encodes encoded state over time
        state_output, self.state_hidden = self.state_gru(obs_encoded, self.state_hidden)

        # critic value of the current state
        value = self.critic(state_output).squeeze()
        observation_hidden = self.state_hidden.squeeze(0)

        # Embed and encode commands
        output = self.tokenizer.encode_commands(commands)
        cmd_hidden = hidden.permute(1, 0, 2).reshape(hidden.shape[1], -1) if hidden.shape[0] == 2 else hidden

        # concatenate the encoding of the state with every encoded command individually
        observation_hidden = torch.stack([observation_hidden.squeeze()] * len(output)
        cmd_selector_input = torch.cat([cmd_hidden, observation_hidden], -1)

        # compute a score for each of the commands
        score = self.att_cmd(cmd_selector_input).squeeze()
        if len(score.shape) == 0:
            # if only one admissible_command
            score = score.unsqueeze(0)
        prob = F.softmax(score, dim=0)

        # sample from the distribution over commands
        index = prob.multinomial(num_samples=1).squeeze()
        action = command_strings[index]

        return score, prob, value, action, index

if __name__ == "__main__":
    model = Model(DEVICE)
