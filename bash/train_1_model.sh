#!/bin/bash
# chmod +x ./bash/train_1_model.sh
# ./bash/train_1_model.sh | tee -a ./log/train_model.log

python3 ./experiments/train_mnist.py --data_path data --output_path results --batch_size 128 --epochs 200
python3 ./experiments/train_cifar.py --data_path data --output_path results --model resnet --batch_size 128 --epochs 200
python3 ./experiments/train_cifar.py --data_path data --output_path results --model vgg --batch_size 128 --epochs 200

python3 ./experiments/train_numeric.py --data banknote --data_path data --output_path results --batch_size 128 --epochs 400
python3 ./experiments/train_numeric.py --data htru2 --data_path data --output_path results --batch_size 128 --epochs 400
python3 ./experiments/train_numeric.py --data breastcancer --data_path data --output_path results --batch_size 256 --epochs 400

# python3 ./experiments/train_numeric.py --data segment --data_path data --output_path results --batch_size 128 --epochs 400
# python3 ./experiments/train_numeric.py --data texture --data_path data --output_path results --batch_size 128 --epochs 400
