import os
import sys

sys.path.append(os.getcwd())
print(*sys.path, sep='\n')

from experiments import sklearn_attack_against_rc

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
                sklearn_attack_against_rc(d, MODEL, a, epsilons=[0], idx=i)


if __name__ == '__main__':
    main()

# Example: Running from terminal
# nohup python3 ./run/benchmark_tree_rc.py > ./log/benchmark_tree_rc.out 2> ./log/benchmark_tree_rc.err & tail -f ./log/benchmark_tree_rc.out
