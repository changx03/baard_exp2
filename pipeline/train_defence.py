import os
import sys

import numpy as np

sys.path.append(os.getcwd())
LIB_PATH = os.getcwd() + "/art_library"
sys.path.append(LIB_PATH)
# print("sys.path ", sys.path)
from defences.magnet import Autoencoder1, Autoencoder2, MagNetAutoencoderReformer, MagNetDetector, MagNetOperator

BATCH_SIZE = 256


def train_magnet(data, model_name, X_train, y_train, X_val, param, device, path, epochs, model=None):
    detectors = []

    if data == 'mnist':
        n_channel = 1
        detectors.append(MagNetDetector(
            encoder=Autoencoder1(n_channel=n_channel),
            classifier=model,
            lr=param['lr'],
            batch_size=BATCH_SIZE,
            weight_decay=param['weight_decay'],
            x_min=0.0,
            x_max=1.0,
            noise_strength=param['noise_strength'],
            algorithm='error',
            p=1,
            device=device))

        detectors.append(MagNetDetector(
            encoder=Autoencoder2(n_channel),
            classifier=model,
            lr=param['lr'],
            batch_size=BATCH_SIZE,
            weight_decay=param['weight_decay'],
            x_min=0.0,
            x_max=1.0,
            noise_strength=param['noise_strength'],
            algorithm='error',
            p=2,
            device=device))
    elif data == 'cifar10':
        n_channel = 3
        autoencoder = Autoencoder2(n_channel=n_channel)
        detectors.append(MagNetDetector(
            encoder=autoencoder,
            classifier=model,
            lr=param['lr'],
            batch_size=BATCH_SIZE,
            weight_decay=param['weight_decay'],
            x_min=0.0,
            x_max=1.0,
            noise_strength=param['noise_strength'],
            algorithm='error',
            p=2,
            device=device))
        detectors.append(MagNetDetector(
            encoder=autoencoder,
            classifier=model,
            lr=param['lr'],
            batch_size=BATCH_SIZE,
            weight_decay=param['weight_decay'],
            x_min=0.0,
            x_max=1.0,
            noise_strength=param['noise_strength'],
            algorithm='prob',
            temperature=10,
            device=device))
        detectors.append(MagNetDetector(
            encoder=autoencoder,
            classifier=model,
            lr=param['lr'],
            batch_size=BATCH_SIZE,
            weight_decay=param['weight_decay'],
            x_min=0.0,
            x_max=1.0,
            noise_strength=param['noise_strength'],
            algorithm='prob',
            temperature=40,
            device=device))
    else:
        raise NotImplementedError

    # Load existing autoencoder or train new one
    for i, d in enumerate(detectors, start=1):
        file_ae = os.path.join(path, '{}_{}_magnet_autoencoder_{}.pt'.format(data, model_name, str(i)))
        if os.path.exists(file_ae):
            print('Found existing MagNet autoencoder:', file_ae)
            d.load(file_ae)
        else:
            # cifar10 use the autoencoder 3 times!
            if data == 'mnist' or (data == 'cifar10' and i == 1):
                d.fit(X_train, y_train, epochs=epochs)
            mse = d.score(X_val)
            print('MSE training set: {:.6f}, validation set: {:.6f}'.format(
                d.history_train_loss[-1] if len(d.history_train_loss) > 0 else np.inf, mse))
            d.search_threshold(X_val, fp=param['fp'], update=True)
            print('Threshold:', d.threshold)
            d.save(file_ae)
            print('Saved to:', file_ae)

    reformer = MagNetAutoencoderReformer(
        encoder=detectors[0].encoder,
        batch_size=BATCH_SIZE,
        device=device)

    detector = MagNetOperator(
        classifier=model,
        detectors=detectors,
        reformer=reformer,
        batch_size=BATCH_SIZE,
        device=device)
    return detector
