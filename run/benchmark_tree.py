import os
import sys

sys.path.append(os.getcwd())
print(*sys.path, sep='\n')

from experiments import sklearn_attack_against_baard
from experiments import sklearn_attack_against_rc

MODEL = 'tree'

params = []
for b in range(1, 4):
    path = os.path.join('params', 'baard_tune_{}s.json'.format(b))
    params.append(path)


def main():
    attacks = ['tree', 'boundary']
    for i in range(5):
        for d in ['banknote', 'breastcancer', 'htru2']:
            for a in attacks:
                sklearn_attack_against_rc(d, MODEL, a, epsilons=[0], idx=i)
                for p in params:
                    sklearn_attack_against_baard(d, MODEL, a, epsilons=[0], idx=i, baard_param=p)


if __name__ == '__main__':
    main()
    # sklearn_attack_against_rc('banknote', 'tree', 'boundary', epsilons=[0], idx=i)
    # sklearn_attack_against_baard('banknote', MODEL, 'boundary', epsilons=[0], idx=0, baard_param=params[-1])
