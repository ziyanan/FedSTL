#!/bin/sh

python3.9 main_fedavg.py --model GRU --method FedAvg --max_lr 0.001 --mode train
python3.9 main_fedavg.py --model RNN --method FedAvg --max_lr 0.001 --mode train

python3.9 main_fedprox.py --model GRU --method FedProx --max_lr 0.001 --mode train
python3.9 main_fedprox.py --model RNN --method FedProx --max_lr 0.001 --mode train

python3.9 main_ditto.py --model GRU --method Ditto --max_lr 0.001 --mode train
python3.9 main_ditto.py --model RNN --method Ditto --max_lr 0.001 --mode train

python3.9 main_fedrep.py --model GRU --method FedRep --max_lr 0.001 --mode train
python3.9 main_fedrep.py --model RNN --method FedRep --max_lr 0.001 --mode train

python3.9 main_ifca.py --model GRU --method IFCA --max_lr 0.001 --mode train
python3.9 main_ifca.py --model RNN --method IFCA --max_lr 0.001 --mode train