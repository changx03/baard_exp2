import torch
import torch.nn as nn
import torch.nn.functional as F


class NumericModel(nn.Module):
    def __init__(self, n_features, n_hidden, n_classes, use_prob=True):
        super(NumericModel, self).__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.use_prob = use_prob

        self.ln1 = nn.Linear(n_features, n_hidden)
        self.ln2 = nn.Linear(n_hidden, n_hidden)
        self.ln3 = nn.Linear(n_hidden, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def before_softmax(self, x):
        x = F.relu(self.ln1(x))
        x = F.relu(self.ln2(x))
        x = self.ln3(x)
        return x

    def forward(self, x):
        x = self.before_softmax(x)
        if self.use_prob:
            x = self.softmax(x)
        return x


def test(model):
    model.to('cpu')
    X = torch.randn(5, 4, dtype=torch.float32)
    y = model(X)
    print(y.size())


if __name__ == '__main__':
    test(NumericModel(4, 16, 3))
