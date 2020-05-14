#!/usr/bin/env bash
lfss=0.08
CUDA_VISIBLE_DEVICES=0 python vae_mnist_eval_new.py --part 0 --total 8 --lfss $lfss &> output0.log &
CUDA_VISIBLE_DEVICES=1 python vae_mnist_eval_new.py --part 1 --total 8 --lfss $lfss &> output1.log &
CUDA_VISIBLE_DEVICES=2 python vae_mnist_eval_new.py --part 2 --total 8 --lfss $lfss &> output2.log &
CUDA_VISIBLE_DEVICES=3 python vae_mnist_eval_new.py --part 3 --total 8 --lfss $lfss &> output3.log &
CUDA_VISIBLE_DEVICES=4 python vae_mnist_eval_new.py --part 4 --total 8 --lfss $lfss &> output4.log &
CUDA_VISIBLE_DEVICES=5 python vae_mnist_eval_new.py --part 5 --total 8 --lfss $lfss &> output5.log &
CUDA_VISIBLE_DEVICES=6 python vae_mnist_eval_new.py --part 6 --total 8 --lfss $lfss &> output6.log &
CUDA_VISIBLE_DEVICES=7 python vae_mnist_eval_new.py --part 7 --total 8 --lfss $lfss &> output7.log &
