#!/bin/bash
#regression
#freesolv
python main_graph_supcon.py --lr_decay_epochs 800 --lr_decay_rate 0.5 --learning_rate 0.001 --batch_size 128 --epochs 800 \
        --trial 1 --dataset freesolv --num_tasks 1 --temp 0.07 --gamma1 4 --gamma2 1 --threshold 0.8 --mlp_layers 1 \
        --wscl 1 --wrecon 1 --data_folder "datasets" --num_gc_layers 3

:<<!
#esol
python main_graph_supcon.py --lr_decay_epochs 800 --lr_decay_rate 0.5 --learning_rate 0.001 --batch_size 128 --epochs 800 \
	        --trial 1 --dataset freesolv --num_tasks 1 --temp 0.07 --gamma1 4 --gamma2 1 --threshold 0.8 --mlp_layers 1 \
		        --wscl 1 --wrecon 1 --data_folder "datasets" --num_gc_layers 3
#lipophilicity
python main_graph_supcon.py --lr_decay_epochs 800 --lr_decay_rate 0.5 --learning_rate 0.001 --batch_size 128 --epochs 800 \
	        --trial 1 --dataset freesolv --num_tasks 1 --temp 0.07 --gamma1 4 --gamma2 1 --threshold 0.8 --mlp_layers 1 \
		        --wscl 1 --wrecon 1 --data_folder "datasets" --num_gc_layers 3
!

#classification
#bace
python main_graph_supcon.py --lr_decay_epochs 300 --lr_decay_rate 0.5 --learning_rate 0.001 --batch_size 128 --epochs 300 \
        --trial 1 --dataset bace --num_tasks 1 --temp 0.07 --mlp_layers 2 --classification\
        --wscl 1 --wrecon 0.1 --data_folder "datasets" --num_gc_layers 7

:<<!
#tox21
python main_graph_supcon.py --lr_decay_epochs 300 --lr_decay_rate 0.5 --learning_rate 0.001 --batch_size 128 --epochs 300 \
	        --trial 1 --dataset tox21 --num_tasks 12 --temp 0.07 --mlp_layers 2 --classification\
		        --wscl 1 --wrecon 0.1 --data_folder "datasets" --num_gc_layers 7

#sider
python main_graph_supcon.py --lr_decay_epochs 300 --lr_decay_rate 0.5 --learning_rate 0.001 --batch_size 128 --epochs 300 \
	        --trial 1 --dataset sider --num_tasks 27 --temp 0.07 --mlp_layers 2 --classification\
		        --wscl 1 --wrecon 0.1 --data_folder "datasets" --num_gc_layers 7
!
#clintox
python main_graph_supcon.py --lr_decay_epochs 300 --lr_decay_rate 0.5 --learning_rate 0.001 --batch_size 128 --epochs 300 \
	        --trial 1 --dataset clintox --num_tasks 2 --temp 0.07 --mlp_layers 2 --classification\
		        --wscl 1 --wrecon 0.1 --data_folder "datasets" --num_gc_layers 7
