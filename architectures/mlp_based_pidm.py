import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyEmbedding(nn.Module):
    def __init__(self, state_dim, **kwargs):
        super(PolicyEmbedding, self).__init__()
        self.state_dim = state_dim

        self.linear1 = nn.Linear(state_dim, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, 512)
        self.linear5 = nn.Linear(512, 512)

        self.output_dim = 512

    def forward(self, state):
        emb = self.linear1(state)
        emb = F.relu(emb)
        emb = self.linear2(emb)
        emb = F.relu(emb)
        emb = self.linear3(emb)
        emb = F.relu(emb)
        emb = self.linear4(emb)
        emb = F.relu(emb)
        emb = self.linear5(emb)
        emb = F.relu(emb)
        return emb