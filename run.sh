#!/bin/sh

python3.9 main_fedavg.py --model GRU --method FedAvg --max_lr 0.01 --mode train --client_iter 10 --epoch 30
python3.9 main_fedavg.py --model RNN --method FedAvg --max_lr 0.01 --mode train --client_iter 10 --epoch 30
python3.9 main_fedavg.py --model LSTM --method FedAvg --max_lr 0.01 --mode train --client_iter 10 --epoch 30

python3.9 main_ditto.py --mode train --model GRU --method Ditto --max_lr 0.01 --epoch 30 --client_iter 10

python3.9 main_fedrep.py --model GRU --method FedRep --max_lr 0.01 --mode train --client_iter 10
python3.9 main_fedrep.py --model RNN --method FedRep --max_lr 0.01 --mode train --client_iter 10
python3.9 main_fedrep.py --model LSTM --method FedRep --max_lr 0.01 --mode train --client_iter 10

python3.9 main_ifca.py --model GRU --method IFCA --max_lr 0.01 --mode train --client_iter 10
python3.9 main_ifca.py --model RNN --method IFCA --max_lr 0.01 --mode train --client_iter 10
python3.9 main_ifca.py --model LSTM --method IFCA --max_lr 0.01 --mode train --client_iter 10