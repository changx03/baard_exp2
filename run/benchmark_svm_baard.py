import os
import sys

sys.path.append(os.getcwd())
print(*sys.path, sep='\n')

from experiments.sklearn_attack_against_baard import sklearn_attack_against_baard

params = []
for i in range(1, 4):
    path = os.path.join('params', 'baard_num_{}.json'.format(i))
    params.append(path)


def main():
    eps = [0.05, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0]
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

# Example: Running from terminal
# nohup python3 ./run/benchmark_svm_baard.py > ./log/benchmark_svm_baard.out 2> ./log/benchmark_svm_baard.err & tail -f ./log/benchmark_svm_baard.out
