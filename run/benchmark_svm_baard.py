import argparse
import os
import sys

sys.path.append(os.getcwd())
print(*sys.path, sep='\n')

from experiments.sklearn_attack_against_baard import sklearn_attack_against_baard

params = []
for i in range(1, 4):
    path = os.path.join('params', 'baard_num_{}.json'.format(i))
    params.append(path)


def run_inner(i):
    eps = [0.05, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0]
    attacks = ['fgsm', 'bim']
    for d in ['banknote', 'breastcancer', 'htru2']:
        for a in attacks:
            for p in params:
                sklearn_attack_against_baard(d, 'svm', a, epsilons=eps, idx=i, baard_param=p)
        for p in params:
            sklearn_attack_against_baard(d, 'svm', att='boundary', epsilons=[0], idx=i, baard_param=p)


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
# nohup python3 ./run/benchmark_svm_baard.py -n 2 > ./log/benchmark_svm_baard.out 2> ./log/benchmark_svm_baard.err & tail -f ./log/benchmark_svm_baard.out
