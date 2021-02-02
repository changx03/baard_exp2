#!/bin/bash
# chmod +x ./bash/preprocess_baard.sh
# ./bash/preprocess_baard.sh | tee ./log/preprocess_baard.log

python3 ./experiments/preprocess_baard.py --data mnist --model basic --pretrained mnist_200.pt

MODELS=('resnet' 'vgg')
for MOD in "${MODELS[@]}"; do
    python3 ./experiments/preprocess_baard.py --data cifar10 --model $MOD --pretrained "cifar10_"$MOD"_200.pt"
done

python3 ./experiments/preprocess_baard.py --data banknote --model basic16 --pretrained banknote_400.pt
python3 ./experiments/preprocess_baard.py --data breastcancer --model basic120 --pretrained breastcancer_400.pt
python3 ./experiments/preprocess_baard.py --data htru2 --model basic32 --pretrained htru2_400.pt

python3 ./experiments/preprocess_baard_sklearn.py --data banknote --model svm
python3 ./experiments/preprocess_baard_sklearn.py --data breastcancer --model svm
python3 ./experiments/preprocess_baard_sklearn.py --data htru2 --model svm
