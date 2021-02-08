import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset


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
    dataset = TensorDataset(torch.from_numpy(X).type(torch.float32))
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
