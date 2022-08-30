if [ $1 == 'freesolv' ]
then
	python main.py --lr_decay_epochs 800 --lr_decay_rate 0.5 --learning_rate 0.001 --batch_size 128 --epochs 800 \
		--trial 1 --dataset freesolv --num_tasks 1 --temp 0.07 --gamma1 4 --gamma2 1 --threshold 0.8 --mlp_layers 1 \
		--wscl 1 --wrecon 1 --data_dir "datasets" --num_gc_layers 3
elif [ $1 == 'delaney' ]
then
	python main.py --lr_decay_epochs 300 --lr_decay_rate 0.5 --learning_rate 0.001 --batch_size 128 --epochs 300 \
		                --trial 1 --dataset delaney --num_tasks 1 --temp 0.07 --gamma1 2 --gamma2 2 --threshold 0.8 --mlp_layers 2 \
				                        --wscl 1 --wrecon 1 --data_dir "datasets" --num_gc_layers 3
elif [ $1 == 'lipophilicity' ]
then
	python main.py --lr_decay_epochs 800 --lr_decay_rate 0.5 --learning_rate 0.001 --batch_size 128 --epochs 800 \
		                --trial 1 --dataset lipo --num_tasks 1 --temp 0.07 --gamma1 4 --gamma2 1 --threshold 0.8 --mlp_layers 2 \
				                        --wscl 1 --wrecon 1 --data_dir "datasets" --num_gc_layers 3

elif [ $1 == 'bace' ]
then
	python main.py --lr_decay_epochs 300 --lr_decay_rate 0.5 --learning_rate 0.0001 --batch_size 128 --epochs 300 \
		        --trial 1 --dataset bace --num_tasks 1 --temp 0.07 --mlp_layers 1 --classification\
			        --wscl 1 --wrecon 0.1 --data_dir "datasets" --num_gc_layers 7
elif [ $1 == 'tox21' ]
then
	python main.py --lr_decay_epochs 300 --lr_decay_rate 0.5 --learning_rate 0.0001 --batch_size 128 --epochs 300 \
		                --trial 1 --dataset tox21 --num_tasks 12 --temp 0.07 --mlp_layers 1 --classification\
				                        --wscl 1 --wrecon 0.1 --data_dir "datasets" --num_gc_layers 7
elif [ $1 == 'sider' ]
then
	python main.py --lr_decay_epochs 300 --lr_decay_rate 0.5 --learning_rate 0.0001 --batch_size 128 --epochs 300 \
		                --trial 1 --dataset sider --num_tasks 27 --temp 0.07 --mlp_layers 1 --classification\
				                        --wscl 1 --wrecon 0.1 --data_dir "datasets" --num_gc_layers 7
elif [ $1 == 'clintox' ]
then	
	python main.py --lr_decay_epochs 300 --lr_decay_rate 0.5 --learning_rate 0.0001 --batch_size 128 --epochs 100 \
		                --trial 1 --dataset clintox --num_tasks 2 --temp 0.07 --mlp_layers 2 --classification\
				                        --wscl 1 --wrecon 0.1 --data_dir "datasets" --num_gc_layers 7
else
	echo "Input a dataset"
fi
