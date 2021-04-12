import datetime
import os
import sys
import time

sys.path.append(os.getcwd())
LIB_PATH = os.getcwd() + "/art_library"
sys.path.append(LIB_PATH)

import numpy as np
import torch
import torch.nn as nn
from models.torch_util import predict_numpy, train, validate
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from experiments.get_advx_untargeted import get_advx_untargeted

BATCH_SIZE = 192


class SurrogateModel(nn.Module):
    """This is the surrogate model for BAARD"""

    def __init__(self, in_channels=1):
        super(SurrogateModel, self).__init__()
        # Compute nodes after flatten
        if in_channels == 1:
            n_flat = 9216
        elif in_channels == 3:
            n_flat = 12544
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool1 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten(1)
        self.fc1 = nn.Linear(n_flat, 2048)
        self.fc2 = nn.Linear(2048, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_surrogate(model,
                       detector,
                       data_name,
                       X_train,
                       y_train,
                       X_test,
                       y_test,
                       epochs,
                       att_name,
                       epsilons,
                       eps_test,
                       path,
                       device):
    time_start = time.time()

    n_eps = len(epsilons)
    print('[SURROGATE] n_eps:', n_eps)
    n_samples = X_train.shape[0]
    print('[SURROGATE] n_samples:', n_samples)
    n_per_att = n_samples // n_eps
    print('[SURROGATE] n_per_att:', n_per_att)
    n_channels = X_train.shape[1]
    print('[SURROGATE] n_channels:', n_channels)

    # Start building training set
    path_surro_X_train = os.path.join(path, 'surro_X_train.npy')
    path_surro_y_train = os.path.join(path, 'surro_y_train.npy')
    if os.path.exists(path_surro_X_train) and os.path.exists(path_surro_y_train):
        print('[SURROGATE] Found existing:', path_surro_X_train)
        X_train = np.load(path_surro_X_train)
        y_train = np.load(path_surro_y_train)
    else:
        idx_shuffle = np.random.permutation(n_samples)
        X_train = X_train[idx_shuffle]
        y_train = y_train[idx_shuffle]
        np.save(path_surro_X_train, X_train)
        np.save(path_surro_y_train, y_train)
        print('[SURROGATE] Save to:', path_surro_X_train)

    print('[SURROGATE] Generating adversarial examples...')
    path_surro_adv_train = os.path.join(path, 'surro_adv_train.npy')
    if os.path.exists(path_surro_adv_train):
        adv_train = np.load(path_surro_adv_train)
        print('[SURROGATE] Found existing:', path_surro_adv_train)
    else:
        adv_train = np.zeros_like(X_train)
        idx_start = 0
        for e in epsilons:
            idx_end = idx_start + n_per_att
            print('[SURROGATE] Generating {} advx eps={}'.format((idx_end - idx_start), e))
            if e == epsilons[-1]:
                subset = X_train[idx_start:]
            else:
                subset = X_train[idx_start: idx_end]
            adv = get_advx_untargeted(
                model,
                data_name=data_name,
                att_name=att_name,
                eps=e,
                device=device,
                X=subset,
                batch_size=BATCH_SIZE)
            adv_train[idx_start: idx_end] = adv
            idx_start = idx_end
        np.save(path_surro_adv_train, adv_train)
        print('[SURROGATE] Save to:', path_surro_adv_train)

    print('[SURROGATE] Getting labels from BAARD...')
    path_surro_lbl_X_train = os.path.join(path, 'surro_lbl_X_train.npy')
    path_surro_lbl_adv_train = os.path.join(path, 'surro_lbl_adv_train.npy')
    if os.path.exists(path_surro_lbl_X_train) and os.path.exists(path_surro_lbl_adv_train):
        print('[SURROGATE] Found existing:', path_surro_lbl_X_train)
        lbl_X_train = np.load(path_surro_lbl_X_train)
        lbl_adv_train = np.load(path_surro_lbl_adv_train)
    else:
        pred_X_train = predict_numpy(model, X_train, device)
        lbl_X_train = detector.detect(X_train, pred_X_train)
        np.save(path_surro_lbl_X_train, lbl_X_train)
        print('[SURROGATE] Save to:', path_surro_lbl_X_train)

        pred_adv_train = predict_numpy(model, adv_train, device)
        lbl_adv_train = detector.detect(adv_train, pred_adv_train)
        np.save(path_surro_lbl_adv_train, lbl_adv_train)
        print('[SURROGATE] Save to:', path_surro_lbl_adv_train)

    # Combine clean and advx together
    data_train = np.concatenate((X_train, adv_train))
    lbl_train = np.concatenate((lbl_X_train, lbl_adv_train))

    print('[SURROGATE] Creating test set...')
    path_surro_X_test = os.path.join(path, 'surro_X_.npy')
    path_surro_y_test = os.path.join(path, 'surro_y_.npy')
    path_surro_adv_test = os.path.join(path, 'surro_adv_test.npy')
    if not os.path.exists(path_surro_adv_test):
        np.save(path_surro_X_test, X_test)
        np.save(path_surro_y_test, y_test)
        print('[SURROGATE] Save to:', path_surro_X_test)

        adv_test = get_advx_untargeted(
            model,
            data_name=data_name,
            att_name=att_name,
            eps=eps_test,
            device=device,
            X=X_test,
            batch_size=BATCH_SIZE)
        np.save(path_surro_adv_test, adv_test)
        print('[SURROGATE] Save to:', path_surro_adv_test)
    else:
        print('[SURROGATE] Found existing:', path_surro_adv_test)
        adv_test = np.load(path_surro_adv_test)

    print('[SURROGATE] Getting labels for test set...')
    path_lbl_X_test = os.path.join(path, 'surro_lbl_X_test.npy')
    path_lbl_adv_test = os.path.join(path, 'surro_lbl_adv_test.npy')
    if os.path.exists(path_lbl_X_test) and os.path.exists(path_lbl_adv_test):
        print('[SURROGATE] Found existing:', path_lbl_X_test)
        lbl_X_test = np.load(path_lbl_X_test)
        lbl_adv_test = np.load(path_lbl_adv_test)
    else:
        pred_X_test = predict_numpy(model, X_test, device)
        lbl_X_test = detector.detect(X_test, pred_X_test)
        np.save(path_lbl_X_test, lbl_X_test)
        print('[SURROGATE] Save to:', path_lbl_X_test)

        pred_adv_test = predict_numpy(model, adv_test, device)
        lbl_adv_test = detector.detect(adv_test, pred_adv_test)
        np.save(path_lbl_adv_test, lbl_adv_test)
        print('[SURROGATE] Save to:', path_lbl_adv_test)

    data_test = np.concatenate((X_test, adv_test))
    lbl_test = np.concatenate((lbl_X_test, lbl_adv_test))

    print('[SURROGATE] Training surrogate model...')
    surrogate = SurrogateModel(in_channels=n_channels).to(device)
    optimizer = optim.AdamW(surrogate.parameters(), lr=0.001, weight_decay=1e-6)
    loss = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    tensor_data_train = torch.from_numpy(data_train).type(torch.float32)
    tensor_lbl_train = torch.from_numpy(lbl_train).type(torch.long)
    dataset_train = TensorDataset(tensor_data_train, tensor_lbl_train)
    loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)

    tensor_data_test = torch.from_numpy(data_test).type(torch.float32)
    tensor_lbl_test = torch.from_numpy(lbl_test).type(torch.long)
    dataset_test = TensorDataset(tensor_data_test, tensor_lbl_test)
    loader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)

    hist_train_loss = []
    for e in range(epochs):
        train_start = time.time()
        tr_loss, tr_acc = train(surrogate, loader_train, loss, optimizer, device)
        va_loss, va_acc = validate(surrogate, loader_test, loss, device)
        scheduler.step()
        time_elapsed = time.time() - train_start
        print(('[SURROGATE] {:2d}/{:d}[{:s}] Train Loss: {:.4f} Acc: {:.2f}%, Test Loss: {:.4f} Acc: {:.2f}%').format(
            e + 1, epochs, str(datetime.timedelta(seconds=time_elapsed)), tr_loss, tr_acc * 100, va_loss, va_acc * 100))
        hist_train_loss.append(tr_loss)
        if len(hist_train_loss) > 10 and hist_train_loss[-10] <= tr_loss:
            print('[SURROGATE] Training is converged at:', e)
            break
    time_elapsed = time.time() - time_start
    print('[SURROGATE] Total training time:', str(datetime.timedelta(seconds=time_elapsed)))
    return surrogate


# Testing
if __name__ == '__main__':
    x = torch.randn((64, 3, 32, 32))
    net = SurrogateModel(in_channels=3)
    out = net(x)
    print(out.size())
    print(out[:5])
