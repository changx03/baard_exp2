import os
import sys

sys.path.append(os.getcwd())
print(*sys.path, sep='\n')

from experiments import sklearn_attack_against_rc


def main():
    eps = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    attacks = ['fgsm', 'bim']
    for i in range(5):
        for d in ['banknote', 'breastcancer', 'htru2']:
            for a in attacks:
                sklearn_attack_against_rc(d, 'svm', a, epsilons=eps, idx=i)
            sklearn_attack_against_rc(d, 'svm', 'boundary', epsilons=[0], idx=i)


if __name__ == '__main__':
    main()

# Example: Running from terminal
# nohup python3 ./run/benchmark_svm_rc.py > ./log/benchmark_svm_rc.out 2> ./log/benchmark_svm_rc.err & tail -f ./log/benchmark_svm_rc.out