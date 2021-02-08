#!/bin/bash

python3 ./pipeline/run_whitebox_attack_against_baard.py --data mnist --model dnn --i 2
python3 ./pipeline/run_whitebox_attack_against_baard.py --data mnist --model dnn --i 3
python3 ./pipeline/run_whitebox_attack_against_baard.py --data mnist --model dnn --i 4
