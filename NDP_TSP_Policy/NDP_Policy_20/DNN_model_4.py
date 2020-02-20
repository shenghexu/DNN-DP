import torch
import torch.nn as nn
from pdb import set_trace as bp





class DP_M(nn.Module):
    def __init__(self):
        super(DP_M, self).__init__()
        self.fc1    = nn.Linear(16, 2, bias=False)
        #bp()
        #K = torch.Tensor([[0.0, 1.0, 0.0, 0.0,  0.0,  ], 
        #    [0.0, 1.0, 1.0, 0.0]])
        #K = torch.unsqueeze(K, 0)

        K = torch.zeros(2, 16)
        K[0, 1]     = 1.0
        K[0, 6]     = 1.0
        K[0, 11]    = 1.0
        K[1, 2]     = 1.0
        K[1, 9]     = 1.0
        K[1, 13]    = 1.0

        self.fc1.weight.data = K

    def forward(self, x):
        Batch_size = x.shape[0]
        #bp()
        out = self.fc1(x)
        #bp()
        #out = self.relu(out)
        #out = self.fc2(out)
        #out = self.relu(out)
        #out = self.fc3(out)
        indices = torch.argmin(out, dim=1)
        
        #values = x[0:Batch_size, indices]
        
        return out, indices







