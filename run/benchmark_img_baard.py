import os
import sys

sys.path.append(os.getcwd())
print(*sys.path, sep='\n')

from experiments.pytorch_attack_against_baard_img import pytorch_attack_against_baard

DATA = 'mnist'
MODEL = 'dnn'
params = []
for i in range(1, 4):
    path = os.path.join('params', 'baard_mnist_{}.json'.format(i))
    params.append(path)


def main():
    eps = [0.063, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0]
    attacks = ['fgsm', 'apgd']
    for i in range(5):
        for a in attacks:
            for p in params:
                pytorch_attack_against_baard(DATA, MODEL, a, epsilons=eps, idx=i, baard_param=p, fresh_att=False, fresh_def=False)


if __name__ == '__main__':
    main()

# Example: Running from terminal
# nohup python3 ./run/benchmark_img_baard.py > ./log/benchmark_img_baard.out 2> ./log/benchmark_img_baard.err & tail -f ./log/benchmark_img_baard.out
