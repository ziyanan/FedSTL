"""
The main file for evaluations.
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9

import numpy as np
import torch
from torch.utils.data import DataLoader
from IoTData import SequenceDataset
from utils.update import LocalUpdate, LocalUpdateProp, compute_cluster_id, cluster_id_property, cluster_explore
from utils_training import get_device, to_device, save_model, get_client_dataset, get_shared_dataset, model_init
import sys
import os
import copy
from tqdm import tqdm
from options import args_parser
from network import ShallowRegressionLSTM, ShallowRegressionGRU, ShallowRegressionRNN, MultiRegressionLSTM, MultiRegressionGRU, MultiRegressionRNN
from transformer import TimeSeriesTransformer
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

import matplotlib.pyplot as plt




def main():
    args = args_parser()
    args.device = get_device()
    
    client_dataset = {}
    for c in range(args.client):
        client_dataset[c] = {}
        train_loader_private, trainset_shared, val_loader, test_loader, dataset_len = get_shared_dataset(c, args.dataset)
        client_dataset[c]["train_private"] = train_loader_private
        client_dataset[c]["train_shared"] = trainset_shared
        client_dataset[c]["val"] = val_loader
        client_dataset[c]["test"] = test_loader
        client_dataset[c]["len"] = dataset_len
    print("Loaded client dataset.")

    ############################
    # loading shared model
    glob_model, clust_weight_keys = model_init(args)
    glob_model = to_device(glob_model, args.device)
    net_keys = [*glob_model.state_dict().keys()]

    ############################
    # generate list of local models for each user
    local_weights = {}      # client ID: weight: val
    for c in range(args.client):
        w_local_dict = {}
        for key in glob_model.state_dict().keys():
            w_local_dict[key] = glob_model.state_dict()[key]
        local_weights[c] = w_local_dict
    print("Loaded client models.")

    ############################
    # evaluation on SUMO dataset.
    if args.mode == "eval-sumo":
        print("Testing IFCA")
        model = MultiRegressionRNN(input_dim=6, batch_size=args.batch_size, time_steps=40, sequence_len=10, hidden_dim=16)
        model_path = "/hdd/traffic_data_2019/run-client-20/sumo_10_RNN_glob_cluster_epoch_30.pt"
        model.load_state_dict(torch.load(model_path))
        model.eval()
        local_loss = []     
        local_rho = []
        for c in range(args.client):
            local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c)      
            w_local, loss, cons_loss, idx, rho_perc = local.test(net=model.to(args.device), idx=c, w_glob_keys=None, rho=True)    
            local_loss.append(copy.deepcopy(loss))
            local_rho.append(copy.deepcopy(rho_perc.item()))
            print(idx, loss, rho_perc)
        print(sum(local_loss)/len(local_loss))
        print(sum(local_rho)/len(local_rho))


        print("Testing FedSTL")
        model = MultiRegressionRNN(input_dim=6, batch_size=args.batch_size, time_steps=40, sequence_len=10, hidden_dim=16)
        model_path = "/hdd/traffic_data_2019/run-client-20/sumo_10_RNN_glob_cluster_corr_epoch_30.pt"
        model.load_state_dict(torch.load(model_path))
        model.eval()
        local_loss = []     
        local_rho = []
        for c in range(args.client):
            local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c)      
            w_local, loss, cons_loss, idx, rho_perc = local.test(net=model.to(args.device), idx=c, w_glob_keys=None, rho=True)    
            local_loss.append(copy.deepcopy(loss))
            local_rho.append(copy.deepcopy(rho_perc.item()))
            print(idx, loss, rho_perc)
        print(sum(local_loss)/len(local_loss))
        print(sum(local_rho)/len(local_rho))


        print("Testing IFCA-c")
        model = MultiRegressionRNN(input_dim=6, batch_size=args.batch_size, time_steps=40, sequence_len=10, hidden_dim=16)
        local_loss = []
        local_rho = []
        for c in range(args.client):
            model_path = "/hdd/traffic_data_2019/run-client-20/sumo_10_RNN_client_"+str(c)+"_epoch_30.pt"
            model.load_state_dict(torch.load(model_path))
            model.eval()
            local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c)      
            w_local, loss, cons_loss, idx, rho_perc = local.test(net=model.to(args.device), idx=c, w_glob_keys=None, rho=True)              
            local_loss.append(copy.deepcopy(loss))
            local_rho.append(copy.deepcopy(rho_perc.item()))
            print(idx, loss, rho_perc)
        print(sum(local_loss)/len(local_loss))
        print(sum(local_rho)/len(local_rho))


        print("Testing FedSTL-c")
        model = MultiRegressionRNN(input_dim=6, batch_size=args.batch_size, time_steps=40, sequence_len=10, hidden_dim=16)
        local_loss = []    
        local_rho = [] 
        for c in range(args.client):
            model_path = "/hdd/traffic_data_2019/run-client-20/sumo_10_RNN_client_corr_"+str(c)+"_epoch_30.pt"
            model.load_state_dict(torch.load(model_path))
            model.eval()
            local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c)   
            w_local, loss, cons_loss, idx, rho_perc = local.test(net=model.to(args.device), idx=c, w_glob_keys=None, rho=True) 
            local_loss.append(copy.deepcopy(loss))
            local_rho.append(copy.deepcopy(rho_perc.item()))
            print(idx, loss, rho_perc)
        print(sum(local_loss)/len(local_loss))
        print(sum(local_rho)/len(local_rho))



    elif args.mode == "eval":
        print("IFCA client")
        model = ShallowRegressionLSTM(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
        local_loss = []     
        local_rho = []
        for c in range(args.client):
            model_path = "/hdd/traffic_data_2019/run-model-backup/run/fhwaLSTM_client_"+str(c)+"_epoch_50.pt"
            model.load_state_dict(torch.load(model_path))
            model.eval()
            local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c)                   
            w_local, loss, cons_loss, idx, rho_perc = local.test(net=model.to(args.device), idx=c, w_glob_keys=None, rho=True) 
            local_loss.append(copy.deepcopy(loss))
            local_rho.append(copy.deepcopy(rho_perc.item()))
            print(idx, loss, rho_perc)
        print(sum(local_loss)/len(local_loss))
        print(sum(local_rho)/len(local_rho))
        

        print("FedSTL client")
        model = ShallowRegressionLSTM(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
        local_loss = []    
        local_rho = []
        for c in range(args.client):
            model_path = "/hdd/traffic_data_2019/run/fhwa_10_LSTM_client_eventually_"+str(c)+"_epoch_50.pt"
            model.load_state_dict(torch.load(model_path))
            model.eval()
            local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c)                   # init local update modules
            w_local, loss, cons_loss, idx, rho_perc = local.test(net=model.to(args.device), idx=c, w_glob_keys=None, rho=True) # test local
            local_loss.append(copy.deepcopy(loss))
            local_rho.append(copy.deepcopy(rho_perc.item()))
            print(idx, loss, rho_perc)
        print(sum(local_loss)/len(local_loss))
        print(sum(local_rho)/len(local_rho))


        print("FedAvg")
        model_path = "/hdd/traffic_data_2019/run-model-backup/run/fhwa_RNN_glob_epoch_50.pt"
        model = ShallowRegressionRNN(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
        model = to_device(model, args.device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        local_loss = []
        local_rho = []
        for c in range(args.client):
            local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c)                   # init local update modules
            w_local, loss, cons_loss, idx, rho_perc = local.test(net=model.to(args.device), idx=c, w_glob_keys=None, rho=True) # train local
            local_loss.append(copy.deepcopy(loss))
            local_rho.append(copy.deepcopy(rho_perc.item()))
            print(idx, loss, rho_perc)
        print(sum(local_loss)/len(local_loss))
        print(sum(local_rho)/len(local_rho))


        print("IFCA glob")
        model_path = "/hdd/traffic_data_2019/run-model-backup/run/fhwaLSTM_glob_cluster_epoch_50.pt"
        model = ShallowRegressionLSTM(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
        model = to_device(model, args.device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        local_loss = []
        local_rho = []
        for c in range(args.client):
            local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c)                   # init local update modules
            w_local, loss, cons_loss, idx, rho_perc = local.test(net=model.to(args.device), idx=c, w_glob_keys=None, rho=True) # train local
            local_loss.append(copy.deepcopy(loss))
            local_rho.append(copy.deepcopy(rho_perc.item()))
            print(idx, loss, rho_perc)
        print(sum(local_loss)/len(local_loss))
        print(sum(local_rho)/len(local_rho))

        print("FedSTL glob")
        model_path = "/hdd/traffic_data_2019/run/fhwa_10_LSTM_glob_cluster_eventually_epoch_50.pt"
        model = ShallowRegressionLSTM(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
        model = to_device(model, args.device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        local_loss = []
        local_rho = []
        for c in range(args.client):
            local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c)                   # init local update modules
            w_local, loss, cons_loss, idx, rho_perc = local.test(net=model.to(args.device), idx=c, w_glob_keys=None, rho=True) # train local
            local_loss.append(copy.deepcopy(loss))
            local_rho.append(copy.deepcopy(rho_perc.item()))
            print(idx, loss, rho_perc)
        print(sum(local_loss)/len(local_loss))
        print(sum(local_rho)/len(local_rho))




if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    
    finally:
        print('\nDone.')
        # sys.stdout.close()
        # sys.stdout=stdoutOrigin