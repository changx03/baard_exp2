import argparse
import os
import sys

sys.path.append(os.getcwd())
print(*sys.path, sep='\n')

from experiments.sklearn_attack_against_baard import sklearn_attack_against_baard


def run(i):
    eps = [0.05, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0]
    attacks = ['fgsm', 'bim']
    for d in ['banknote', 'breastcancer', 'htru2']:
        sklearn_attack_against_baard(d, 'svm', att='boundary', epsilons=[0], idx=i)

        for a in attacks:
            sklearn_attack_against_baard(d, 'svm', a, epsilons=eps, idx=i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--idx', type=int, default=0)
    args = parser.parse_args()
    print('args:', args)
    run(args.idx)

# Example: Running from terminal
# nohup python3 ./run/benchmark_svm_baard.py -i 0 > ./log/benchmark_svm_baard_0.out &
