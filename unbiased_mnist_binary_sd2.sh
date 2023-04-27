
python3 main_unbiased.py --dataset mnist --model mnistnet_binary --dp_mechanism no_dp --iid 1 --seed 10263

python3 main_unbiased.py --dataset mnist --model mnistnet_binary --dp_mechanism rr --dp_flip 0.1 --iid 1 --seed 10263

python3 main_unbiased.py --dataset mnist --model mnistnet_binary --dp_mechanism rr --dp_flip 0.2 --iid 1 --seed 10263

python3 main_unbiased.py --dataset mnist --model mnistnet_binary --dp_mechanism rr --dp_flip 0.3 --iid 1 --seed 10263

python3 main_unbiased.py --dataset mnist --model mnistnet_binary --dp_mechanism rr --dp_flip 0.4 --iid 1 --seed 10263

python3 main_unbiased.py --dataset mnist --model mnistnet_binary --dp_mechanism rr --dp_flip 0.5 --iid 1 --seed 10263

#python3 main.py --dataset mnist --model mnist_binary --dp_mechanism rr --dp_flip 0.5