#!/bin/sh

python3.9 fed_train.py --model RNN --mode train-logic --epoch 30 --client_iter 15 --local_updates 10
python3.9 fed_train.py --model RNN --mode train --epoch 30 --client_iter 10 --local_updates 10

python3.9 fed_train.py --model GRU --mode train-logic --epoch 30 --client_iter 15 --local_updates 10
python3.9 fed_train.py --model GRU --mode train --epoch 30 --client_iter 10 --local_updates 10

python3.9 fed_train.py --model LSTM --mode train-logic --epoch 30 --client_iter 15 --local_updates 10
python3.9 fed_train.py --model LSTM --mode train --epoch 30 --client_iter 10 --local_updates 10
