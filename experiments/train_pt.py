import torch


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
