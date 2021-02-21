import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryClassificationModel(nn.Module):
    def __init__(self, n_features, n_hidden):
        super(BinaryClassificationModel, self).__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden

        self.ln1 = nn.Linear(n_features, n_hidden)
        self.ln2 = nn.Linear(n_hidden, n_hidden)
        self.ln3 = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = F.relu(self.ln2(x))
        x = self.ln3(x)
        x = torch.sigmoid(x)
        return x


# Testing
if __name__ == '__main__':
    X = torch.randn((3, 4), dtype=torch.float32)
    net = BinaryClassificationModel(4, 4 * 4)
    y = net(X)
    print(y)
