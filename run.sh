#!/bin/sh

python3.9 main_fedavg.py --model LSTM --method FedAvg --max_lr 0.01
python3.9 main_fedavg.py --model GRU --method FedAvg --max_lr 0.001
python3.9 main_fedavg.py --model RNN --method FedAvg --max_lr 0.001

python3.9 main_fedprox.py --model LSTM --method FedProx --max_lr 0.01
python3.9 main_fedprox.py --model GRU --method FedProx --max_lr 0.001
python3.9 main_fedprox.py --model RNN --method FedProx --max_lr 0.001

python3.9 main_ditto.py --model LSTM --method Ditto --max_lr 0.01
python3.9 main_ditto.py --model GRU --method Ditto --max_lr 0.001
python3.9 main_ditto.py --model RNN --method Ditto --max_lr 0.001