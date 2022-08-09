#!/bin/bash

:<<!
python main_graph_supcon.py --lr_decay_epochs 800 --lr_decay_rate 0.5 --learning_rate 0.001 --batch_size 128 --epochs 800 \
	--trial 1 --dataset freesolv --num_tasks 1 --temp 0.07 --gamma1 1 --gamma2 1 --threshold 0.6 --mlp_layers 1 \
	--wscl 1 --wrecon 1 --data_folder "/home/storage2/yan/code/Drug Property/dataset_handcrafted_moleculeX" --num_gc_layers 3
!

python main_graph_supcon.py --lr_decay_epochs 300 --lr_decay_rate 0.5 --learning_rate 0.001 --batch_size 128 --epochs 300 \
        --trial 1 --dataset tox21 --num_tasks 12 --temp 0.07 --threshold 0.6 --mlp_layers 2 --classification\
        --wscl 1 --wrecon 0.1 --data_folder "/home/storage2/yan/code/Drug Property/dataset_handcrafted_moleculeX" --num_gc_layers 7

