import torch
import torch.nn as nn





class DP_M(nn.Module):
    def __init__(self, N_node):
        super(DP_M, self).__init__()
        self.N_node = N_node
        self.size_element = self.N_node*self.N_node
        self.fc1    = nn.Linear(self.size_element, self.size_element*2)
        self.bn1    = nn.BatchNorm1d(self.size_element*2)
        self.relu   = nn.ReLU()
        self.fc2    = nn.Linear(self.size_element*2, self.size_element*4)
        self.bn2    = nn.BatchNorm1d(self.size_element*4)
        self.fc3    = nn.Linear(self.size_element*4, self.N_node*16)
        self.bn3    = nn.BatchNorm1d(self.N_node*16)
        self.fc_out    = nn.Linear(self.N_node*16, self.N_node-2)

    def forward(self, x):
        Batch_size = x.shape[0]
        x = x/1.4*2.0-1.0
        out = self.fc1(x)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.bn2(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.bn3(out)
        out = self.fc_out(out)
        
        indices = torch.argmax(out, dim=1)

        return out, indices







