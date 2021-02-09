import os
import sys

sys.path.append(os.getcwd())
print(*sys.path, sep='\n')

from experiments import sklearn_attack_against_baard

params = []
for i in range(1, 4):
    path = os.path.join('params', 'baard_tune_{}s.json'.format(i))
    params.append(path)


def main():
    eps = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    attacks = ['fgsm', 'bim']
    for i in range(5):
        for d in ['banknote', 'breastcancer', 'htru2']:
            for a in attacks:
                for p in params:
                    sklearn_attack_against_baard(d, 'svm', a, epsilons=eps, idx=i, baard_param=p)
            for p in params:
                sklearn_attack_against_baard(d, 'svm', att='boundary', epsilons=[0], idx=i, baard_param=p)


if __name__ == '__main__':
    main()
