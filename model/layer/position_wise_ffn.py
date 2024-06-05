import torch.nn as nn


class PositionWiseFFN(nn.Module):

    def __init__(self, d, hidden, drop_out):
        super(PositionWiseFFN, self).__init__()

        self.l1 = nn.Linear(d, hidden)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(drop_out)
        self.l2 = nn.Linear(hidden, d)

    def forward(self, x):
        x = self.dropout(self.activation(self.l1(x)))
        return self.l2(x)