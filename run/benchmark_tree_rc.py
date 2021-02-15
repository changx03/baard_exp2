import argparse
import os
import sys

sys.path.append(os.getcwd())
print(*sys.path, sep='\n')

from experiments.sklearn_attack_against_rc import sklearn_attack_against_rc

MODEL = 'tree'

params = []
for b in range(1, 4):
    path = os.path.join('params', 'baard_num_{}.json'.format(b))
    params.append(path)


def run_inner(i):
    attacks = ['tree', 'boundary']
    datasets = ['banknote', 'breastcancer', 'htru2']
    for d in datasets:
        for a in attacks:
            sklearn_attack_against_rc(d, MODEL, a, epsilons=[0], idx=i)


def run(n):
    for i in range(n):
        run_inner(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_times', type=int, default=1)
    args = parser.parse_args()
    print('args:', args)
    run(args.n_times)

# Example: Running from terminal
# nohup python3 ./run/benchmark_tree_rc.py -n 2 > ./log/benchmark_tree_rc.out 2> ./log/benchmark_tree_rc.err & tail -f ./log/benchmark_tree_rc.out
