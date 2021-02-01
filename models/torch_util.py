import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset


class AddGaussianNoise(torch.nn.Module):
    """Add Gaussian Noise to a tensor"""

    def __init__(self, mean=0., std=1., eps=0.025, x_min=0., x_max=1.):
        super().__init__()
        self.mean = mean
        self.std = std
        self.eps = eps
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        x_noisy = x + self.eps * (torch.randn(x.size()) * self.std + self.mean)
        x_noisy = torch.clip(x_noisy, self.x_min, self.x_max)
        return x_noisy

    def __repr__(self):
        return self.__class__.__name__ + '(mean={}, std={}, eps={})'.format(self.mean, self.std, self.eps)


def predict(model, loader, device):
    model.eval()
    batch = next(iter(loader))
    y = model(batch[0].to(device))
    shape_output = (len(loader.dataset), y.size(1))
    outputs = torch.zeros(shape_output, dtype=torch.float32)

    start = 0
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            end = start + x.size(0)
            outputs[start:end] = model(x)
            start = end

    return outputs.max(1)[1].type(torch.long)


def predict_numpy(model, X, device):
    model.eval()
    dataset = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    tensor_pred = -torch.ones(len(X), dtype=torch.long)

    start = 0
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            end = start + x.size(0)
            outputs = model(x)
            tensor_pred[start:end] = outputs.max(1)[1].type(torch.long).cpu().detach()
            start = end
    return tensor_pred.detach().numpy()


def print_acc_per_label(model, X, y, device):
    labels = np.unique(y)
    for i in labels:
        idx = np.where(y == i)[0]
        x_subset = X[idx]
        y_subset = y[idx]
        pred = predict_numpy(model, x_subset, device)
        correct = np.sum(pred == y_subset)
        print('[{}] {}/{}'.format(i, correct, len(x_subset)))


def train(model, loader, loss, optimizer, device):
    model.train()
    running_loss = 0.0
    corrects = .0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        l = loss(outputs, y)
        l.backward()
        optimizer.step()

        # for display
        running_loss += l.item() * x.size(0)
        preds = outputs.max(1, keepdim=True)[1]
        corrects += preds.eq(y.view_as(preds)).sum().item()

    n = len(loader.dataset)
    epoch_loss = running_loss / n
    epoch_acc = corrects / n
    return epoch_loss, epoch_acc


def validate(model, loader, loss, device):
    model.eval()
    running_loss = .0
    corrects = .0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            l = loss(outputs, y)
            running_loss += l.item() * x.size(0)
            preds = outputs.max(1, keepdim=True)[1]
            corrects += preds.eq(y.view_as(preds)).sum().item()

    n = len(loader.dataset)
    epoch_loss = running_loss / n
    epoch_acc = corrects / n
    return epoch_loss, epoch_acc
