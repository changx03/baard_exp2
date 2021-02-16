import argparse
import os
import sys

sys.path.append(os.getcwd())
print(*sys.path, sep='\n')

from experiments.sklearn_attack_against_baard import sklearn_attack_against_baard

MODEL = 'tree'


def run(i):
    attacks = ['tree', 'boundary']
    datasets = ['banknote', 'breastcancer', 'htru2']
    for d in datasets:
        for a in attacks:
            sklearn_attack_against_baard(d, MODEL, a, epsilons=[0], idx=i, baard_param=p)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--idx', type=int, default=1)
    args = parser.parse_args()
    print('args:', args)
    run(args.idx)

# Example: Running from terminal
# nohup python3 ./run/benchmark_tree_baard.py -i 0 > ./log/benchmark_tree_baard_0.out &
