import argparse
import json
import os

import torch


def check_baard_threshold(file, save=False):
    obj = torch.load(file)
    print('quantiles:', obj['quantiles'])
    print('ks:', obj['ks'])

    if save:
        file_json = os.path.splitext(file)[0] + '.json'
        obj_json = {
            'quantiles': obj['quantiles'].tolist(),
            'ks': obj['ks'].tolist()}
        thresholds = obj['thresholds']
        for i, t in enumerate(thresholds):
            obj_json['threshold_' + str(i)] = t.tolist()
        with open(file_json, 'w') as j:
            json.dump(obj_json, j)
        print('Save to:', file_json)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--save', type=bool, default=False)
    args = parser.parse_args()
    print(args)

    check_baard_threshold(args.file, args.save)
