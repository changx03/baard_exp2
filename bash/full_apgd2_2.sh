#!/bin/bash

OUTPUT_PATHS=("result_1" "result_2" "result_3" "result_4" "result_5")
RANDOM_STATES=(91112 82223 73334 64445 55556)

for i in "${!OUTPUT_PATHS[@]}"; do
    EPOCHS=200
    OUTPUT_PATH="${OUTPUT_PATHS[i]}"
    RANDOM_STATE="${RANDOM_STATES[i]}"
    echo "RUNNING $i..."
    
    echo "Step 1 Train model"
    python3 ./experiments/train_mnist.py --data_path data --output_path $OUTPUT_PATH --batch_size 128 --epochs $EPOCHS --random_state $RANDOM_STATE
    python3 ./experiments/train_cifar.py --data_path data --output_path $OUTPUT_PATH --model resnet --batch_size 128 --epochs $EPOCHS --random_state $RANDOM_STATE
    python3 ./experiments/train_cifar.py --data_path data --output_path $OUTPUT_PATH --model vgg --batch_size 128 --epochs $EPOCHS --random_state $RANDOM_STATE

    echo "Step 2 Train adversarial examples: apgd2 2.0 only"
    python3 ./experiments/train_attacks.py --data mnist --pretrained "mnist_200.pt" --attack apgd2 --eps 2.0 --output_path $OUTPUT_PATH --random_state $RANDOM_STATE
    python3 ./experiments/train_attacks.py --data cifar10 --pretrained "cifar10_resnet_200.pt" --attack apgd2 --eps 2.0 --output_path $OUTPUT_PATH --random_state $RANDOM_STATE
    python3 ./experiments/train_attacks.py --data cifar10 --pretrained "cifar10_vgg_200.pt" --attack apgd2 --eps 2.0 --output_path $OUTPUT_PATH --random_state $RANDOM_STATE

    echo "Step 2.9 Preprocessing training data"
    python3 ./experiments/preprocess_baard.py --data mnist --model basic --pretrained mnist_200.pt --output_path $OUTPUT_PATH --random_state $RANDOM_STATE

    MODELS=('resnet' 'vgg')
    for MOD in "${MODELS[@]}"; do
        python3 ./experiments/preprocess_baard.py --data cifar10 --model $MOD --pretrained "cifar10_"$MOD"_200.pt" --output_path $OUTPUT_PATH --random_state $RANDOM_STATE
    done

    echo "Step 3 Train defences"
    python3 ./experiments/train_defences.py --data mnist --pretrained "mnist_200.pt" --defence baard --param "./params/baard_param_3.json" --suffix 3stage --adv "mnist_basic_apgd2_2.0" --output_path $OUTPUT_PATH --random_state $RANDOM_STATE
    python3 ./experiments/train_defences.py --data cifar10 --pretrained "cifar10_resnet_200.pt" --defence baard --param "./params/baard_param_3.json" --suffix 3stage --adv "cifar10_resnet_apgd2_2.0_adv" --output_path $OUTPUT_PATH --random_state $RANDOM_STATE
    python3 ./experiments/train_defences.py --data cifar10 --pretrained "cifar10_vgg_200.pt" --defence baard --param "./params/baard_param_3.json" --suffix 3stage --adv "cifar10_vgg_apgd2_2.0_adv" --output_path $OUTPUT_PATH --random_state $RANDOM_STATE

    echo "Step 4 Train adversarial examples for training surrogate model"
    python3 ./experiments/train_baard_adv.py --data mnist --model basic --eps 2.0 --test 0 --n_samples 2000 --output_path $OUTPUT_PATH --random_state $RANDOM_STATE
    python3 ./experiments/train_baard_adv.py --data mnist --model basic --eps 2.0 --test 1 --n_samples 2000 --output_path $OUTPUT_PATH --random_state $RANDOM_STATE
    python3 ./experiments/train_baard_adv.py --data cifar10 --model resnet --eps 2.0 --test 0 --n_samples 2000 --output_path $OUTPUT_PATH --random_state $RANDOM_STATE
    python3 ./experiments/train_baard_adv.py --data cifar10 --model resnet --eps 2.0 --test 1 --n_samples 2000 --output_path $OUTPUT_PATH --random_state $RANDOM_STATE
    python3 ./experiments/train_baard_adv.py --data cifar10 --model vgg --eps 2.0 --test 0 --n_samples 2000 --output_path $OUTPUT_PATH --random_state $RANDOM_STATE
    python3 ./experiments/train_baard_adv.py --data cifar10 --model vgg --eps 2.0 --test 1 --n_samples 2000 --output_path $OUTPUT_PATH --random_state $RANDOM_STATE

    echo "Step 4.5 Get labels from BAARD"
    MNIST_TRAIN="mnist_basic_baard_surro_train_eps2.0_size2000.pt"
    MNIST_TEST="mnist_basic_baard_surro_test_eps2.0_size2000.pt"
    CIFAR10_RESNET_TRAIN="cifar10_resnet_baard_surro_train_eps2.0_size2000.pt"
    CIFAR10_RESNET_TEST="cifar10_resnet_baard_surro_test_eps2.0_size2000.pt"
    CIFAR10_VGG_TRAIN="cifar10_vgg_baard_surro_train_eps2.0_size2000.pt"
    CIFAR10_VGG_TEST="cifar10_vgg_baard_surro_test_eps2.0_size2000.pt"

    python3 ./experiments/get_baard_output.py --param ./params/baard_param_3.json --file $MNIST_TRAIN --output_path $OUTPUT_PATH --random_state $RANDOM_STATE
    python3 ./experiments/get_baard_output.py --param ./params/baard_param_3.json --file $MNIST_TEST --output_path $OUTPUT_PATH --random_state $RANDOM_STATE
    python3 ./experiments/get_baard_output.py --param ./params/baard_param_3.json --file $CIFAR10_RESNET_TRAIN --output_path $OUTPUT_PATH --random_state $RANDOM_STATE
    python3 ./experiments/get_baard_output.py --param ./params/baard_param_3.json --file $CIFAR10_RESNET_TEST --output_path $OUTPUT_PATH --random_state $RANDOM_STATE
    python3 ./experiments/get_baard_output.py --param ./params/baard_param_3.json --file $CIFAR10_VGG_TRAIN --output_path $OUTPUT_PATH --random_state $RANDOM_STATE
    python3 ./experiments/get_baard_output.py --param ./params/baard_param_3.json --file $CIFAR10_VGG_TEST --output_path $OUTPUT_PATH --random_state $RANDOM_STATE

    echo "Step 5 Train surrogate model"
    EPOCHS=400

    python3 ./experiments/train_baard_surrogate.py --data mnist --model basic --train $MNIST_TRAIN --test $MNIST_TEST --epochs $EPOCHS --output_path $OUTPUT_PATH --random_state $RANDOM_STATE
    python3 ./experiments/train_baard_surrogate.py --data cifar10 --model resnet --train $CIFAR10_RESNET_TRAIN --test $CIFAR10_RESNET_TEST --epochs $EPOCHS --output_path $OUTPUT_PATH --random_state $RANDOM_STATE
    python3 ./experiments/train_baard_surrogate.py --data cifar10 --model vgg --train $CIFAR10_VGG_TRAIN --test $CIFAR10_VGG_TEST --epochs $EPOCHS --output_path $OUTPUT_PATH --random_state $RANDOM_STATE
    echo
done
