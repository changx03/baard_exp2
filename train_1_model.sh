#!/bin/bash
# chmod +x ./train_1_model.sh

python3 ./experiments/train_mnist.py --data_path data --model_path results --batch_size 128 --epochs 50