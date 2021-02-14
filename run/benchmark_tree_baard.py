import os
import sys

sys.path.append(os.getcwd())
print(*sys.path, sep='\n')

from experiments import sklearn_attack_against_baard

MODEL = 'tree'

params = []
for b in range(1, 4):
    path = os.path.join('params', 'baard_num_{}.json'.format(b))
    params.append(path)


def main():
    attacks = ['tree', 'boundary']
    datasets = ['banknote', 'breastcancer', 'htru2']
    for i in range(5):
        for d in datasets:
            for a in attacks:
                for p in params:
                    sklearn_attack_against_baard(d, MODEL, a, epsilons=[0], idx=i, baard_param=p)


if __name__ == '__main__':
    main()

# Example: Running from terminal
# nohup python3 ./run/benchmark_tree.py > ./log/benchmark_tree.out 2> ./log/benchmark_tree.err & tail -f ./log/benchmark_tree.out
