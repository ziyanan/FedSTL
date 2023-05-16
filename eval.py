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
from utils.update import LocalUpdate, LocalUpdateProp, compute_cluster_id_eval, cluster_id_property, cluster_explore
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



def get_eval_model(net_path, net_type):
    if net_type == 'GRU':
        model = ShallowRegressionGRU(input_dim=2, batch_size=5, time_steps=96, sequence_len=24, hidden_dim=16)
    elif net_type == 'LSTM':
        model = ShallowRegressionLSTM(input_dim=2, batch_size=5, time_steps=96, sequence_len=24, hidden_dim=16)
    elif net_type == 'RNN':
        model = ShallowRegressionRNN(input_dim=2, batch_size=5, time_steps=96, sequence_len=24, hidden_dim=16)
    model = to_device(model, 'cuda')
    model.load_state_dict(torch.load(net_path))
    return model



def get_dict_keys(cluster_id, idxs_users):
    d = {}
    for i in idxs_users:
        for key, val in cluster_id.items():
            if i in val:
                d[i] = key
    return d



def main():
    args = args_parser()
    args.device = get_device()
    model_path = "/hdd/saved_models/"
    
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
        model_types = ["LSTM", "RNN", "GRU"]

        print("==============================================================")
        for type in model_types:
            print("Evaluating FedSTL on model (client teacher)", type)
            local_loss = []
            local_cons_loss = []
            local_rho = []

            for c in range(args.client):
                net_path = os.path.join(model_path, 'fhwa_'+type+'_FedSTL_Client_'+str(c)+'_epoch_30.pt')
                model = get_eval_model(net_path, type)
                model.eval()
                local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c)
                loss, cons_loss, idx, rho_perc = local.test_teacher(net=model.to(args.device), idx=c, w_glob_keys=None, rho=True)
                local_loss.append(copy.deepcopy(loss))
                local_cons_loss.append(copy.deepcopy(cons_loss))
                local_rho.append(copy.deepcopy(rho_perc.item()))
            
            print("Local loss:")
            std = np.std(local_loss)
            error = 1.96 * std / np.sqrt(len(local_loss))
            print("Mean:", np.mean(local_loss))
            print("Error bar:", error)
            print()

            print("Local cons loss:")
            std = np.std(local_cons_loss)
            error = 1.96 * std / np.sqrt(len(local_cons_loss))
            print("Mean:", np.mean(local_cons_loss))
            print("Error bar:", error)
            print()

            print("Local rho:")
            std = np.std(local_rho)
            error = 1.96 * std / np.sqrt(len(local_rho))
            print("Mean:", np.mean(local_rho))
            print("Error bar:", error)
            print()


        print("==============================================================")
        for type in model_types:
            print("Evaluating FedSTL on model", type)
            local_loss = []
            local_cons_loss = []
            local_rho = []

            cluster_models = []
            for c in range(args.cluster):
                net_path = os.path.join(model_path, 'fhwa_'+type+'_FedSTL_Cluster_'+str(c)+'_epoch_30.pt')
                c_model = get_eval_model(net_path, type)
                c_model.eval()
                cluster_models.append(c_model)

            args.frac = 1
            idxs_users = [i for i in range(args.client)]
            cluster_id = cluster_id_property(cluster_models, client_dataset, args, idxs_users)   # cluster: clients
            client2cluster = get_dict_keys(cluster_id, idxs_users)

            for c in range(args.client):
                net_path = os.path.join(model_path, 'fhwa_'+type+'_FedSTL_Client_'+str(c)+'_epoch_30.pt')
                model = get_eval_model(net_path, type)
                model.load_state_dict(cluster_models[client2cluster[c]].state_dict())
                model.eval()
                local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c)
                loss, cons_loss, idx, rho_perc = local.test(net=model.to(args.device), idx=c, w_glob_keys=None, rho=True)
                local_loss.append(copy.deepcopy(loss))
                local_cons_loss.append(copy.deepcopy(cons_loss))
                local_rho.append(copy.deepcopy(rho_perc.item()))
            
            print("Local loss:")
            std = np.std(local_loss)
            error = 1.96 * std / np.sqrt(len(local_loss))
            print("Mean:", np.mean(local_loss))
            print("Error bar:", error)
            print()

            print("Local cons loss:")
            std = np.std(local_cons_loss)
            error = 1.96 * std / np.sqrt(len(local_cons_loss))
            print("Mean:", np.mean(local_cons_loss))
            print("Error bar:", error)
            print()

            print("Local rho:")
            std = np.std(local_rho)
            error = 1.96 * std / np.sqrt(len(local_rho))
            print("Mean:", np.mean(local_rho))
            print("Error bar:", error)
            print()

        print("==============================================================")
        for type in model_types:
            print("Evaluating IFCA on model", type)
            local_loss = []
            local_cons_loss = []
            local_rho = []
            
            net_path = os.path.join(model_path, 'fhwa_'+type+'_IFCA_epoch_30.pt')
            glob_model = get_eval_model(net_path, type)
            glob_model.eval()

            cluster_models = []
            for c in range(args.cluster):
                net_path = os.path.join(model_path, 'fhwa_'+type+'_IFCA_Cluster_'+str(c)+'_epoch_30.pt')
                c_model = get_eval_model(net_path, type)
                c_model.eval()
                cluster_models.append(c_model)

            args.frac = 1
            idxs_users = [i for i in range(args.client)]
            cluster_id = compute_cluster_id_eval(cluster_models, client_dataset, args, idxs_users)   # cluster: clients
            client2cluster = get_dict_keys(cluster_id, idxs_users)

            for c in range(args.client):
                net_path = os.path.join(model_path, 'fhwa_'+type+'_IFCA_Client_'+str(c)+'_epoch_30.pt')
                model = get_eval_model(net_path, type)
                model.load_state_dict(cluster_models[client2cluster[c]].state_dict())
                model.eval()
                local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c)
                loss, cons_loss, idx, rho_perc = local.test(net=glob_model.to(args.device), idx=c, w_glob_keys=None, rho=True)
                local_loss.append(copy.deepcopy(loss))
                local_cons_loss.append(copy.deepcopy(cons_loss))
                local_rho.append(copy.deepcopy(rho_perc.item()))
            
            print("Local loss:")
            std = np.std(local_loss)
            error = 1.96 * std / np.sqrt(len(local_loss))
            print("Mean:", np.mean(local_loss))
            print("Error bar:", error)
            print()

            print("Local cons loss:")
            std = np.std(local_cons_loss)
            error = 1.96 * std / np.sqrt(len(local_cons_loss))
            print("Mean:", np.mean(local_cons_loss))
            print("Error bar:", error)
            print()

            print("Local rho:")
            std = np.std(local_rho)
            error = 1.96 * std / np.sqrt(len(local_rho))
            print("Mean:", np.mean(local_rho))
            print("Error bar:", error)
            print()



        print("==============================================================")
        for type in model_types:
            print("Evaluating FedRep on model", type)
            local_loss = []
            local_cons_loss = []
            local_rho = []
            for c in range(args.client):
                net_path = os.path.join(model_path, 'fhwa_'+type+'_FedRep_'+str(c)+'_epoch_30.pt')
                model = get_eval_model(net_path, type)
                model.eval()
                local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c)
                loss, cons_loss, idx, rho_perc = local.test(net=model.to(args.device), idx=c, w_glob_keys=None, rho=True)
                local_loss.append(copy.deepcopy(loss))
                local_cons_loss.append(copy.deepcopy(cons_loss))
                local_rho.append(copy.deepcopy(rho_perc.item()))
            
            print("Local loss:")
            std = np.std(local_loss)
            error = 1.96 * std / np.sqrt(len(local_loss))
            print("Mean:", np.mean(local_loss))
            print("Error bar:", error)
            print()

            print("Local cons loss:")
            std = np.std(local_cons_loss)
            error = 1.96 * std / np.sqrt(len(local_cons_loss))
            print("Mean:", np.mean(local_cons_loss))
            print("Error bar:", error)
            print()

            print("Local rho:")
            std = np.std(local_rho)
            error = 1.96 * std / np.sqrt(len(local_rho))
            print("Mean:", np.mean(local_rho))
            print("Error bar:", error)
            print()


        print("==============================================================")
        for type in model_types:
            print("Evaluating Ditto on model", type)
            local_loss = []
            local_cons_loss = []
            local_rho = []
            for c in range(args.client):
                net_path = os.path.join(model_path, 'fhwa_'+type+'_Ditto_'+str(c)+'_epoch_30.pt')
                model = get_eval_model(net_path, type)
                model.eval()
                local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c)
                loss, cons_loss, idx, rho_perc = local.test(net=model.to(args.device), idx=c, w_glob_keys=None, rho=True)
                local_loss.append(copy.deepcopy(loss))
                local_cons_loss.append(copy.deepcopy(cons_loss))
                local_rho.append(copy.deepcopy(rho_perc.item()))
            
            print("Local loss:")
            std = np.std(local_loss)
            error = 1.96 * std / np.sqrt(len(local_loss))
            print("Mean:", np.mean(local_loss))
            print("Error bar:", error)
            print()

            print("Local cons loss:")
            std = np.std(local_cons_loss)
            error = 1.96 * std / np.sqrt(len(local_cons_loss))
            print("Mean:", np.mean(local_cons_loss))
            print("Error bar:", error)
            print()

            print("Local rho:")
            std = np.std(local_rho)
            error = 1.96 * std / np.sqrt(len(local_rho))
            print("Mean:", np.mean(local_rho))
            print("Error bar:", error)
            print()


        print("==============================================================")
        for type in model_types:
            print("Evaluating FedProx on model", type)
            net_path = os.path.join(model_path, 'fhwa_'+type+'_FedProx_epoch_30.pt')
            model = get_eval_model(net_path, type)
            model.eval()
            local_loss = []
            local_cons_loss = []
            local_rho = []
            for c in range(args.client):
                local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c)
                loss, cons_loss, idx, rho_perc = local.test(net=model.to(args.device), idx=c, w_glob_keys=None, rho=True)
                local_loss.append(copy.deepcopy(loss))
                local_cons_loss.append(copy.deepcopy(cons_loss))
                local_rho.append(copy.deepcopy(rho_perc.item()))
                # print(idx, loss, rho_perc.item())
            
            print("Local loss:")
            std = np.std(local_loss)
            error = 1.96 * std / np.sqrt(len(local_loss))
            print("Mean:", np.mean(local_loss))
            print("Error bar:", error)
            print()

            print("Local cons loss:")
            std = np.std(local_cons_loss)
            error = 1.96 * std / np.sqrt(len(local_cons_loss))
            print("Mean:", np.mean(local_cons_loss))
            print("Error bar:", error)
            print()

            print("Local rho:")
            std = np.std(local_rho)
            error = 1.96 * std / np.sqrt(len(local_rho))
            print("Mean:", np.mean(local_rho))
            print("Error bar:", error)
            print()

        print("==============================================================")
        for type in model_types:
            print("Evaluating FedAvg on model", type)
            net_path = os.path.join(model_path, 'fhwa_'+type+'_FedAvg_epoch_30.pt')
            model = get_eval_model(net_path, type)
            model.eval()
            local_loss = []
            local_cons_loss = []
            local_rho = []
            for c in range(args.client):
                local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c)
                loss, cons_loss, idx, rho_perc = local.test(net=model.to(args.device), idx=c, w_glob_keys=None, rho=True)
                local_loss.append(copy.deepcopy(loss))
                local_cons_loss.append(copy.deepcopy(cons_loss))
                local_rho.append(copy.deepcopy(rho_perc.item()))
                # print(idx, loss, rho_perc.item())
            
            print("Local loss:")
            std = np.std(local_loss)
            error = 1.96 * std / np.sqrt(len(local_loss))
            print("Mean:", np.mean(local_loss))
            print("Error bar:", error)
            print()

            print("Local cons loss:")
            std = np.std(local_cons_loss)
            error = 1.96 * std / np.sqrt(len(local_cons_loss))
            print("Mean:", np.mean(local_cons_loss))
            print("Error bar:", error)
            print()

            print("Local rho:")
            std = np.std(local_rho)
            error = 1.96 * std / np.sqrt(len(local_rho))
            print("Mean:", np.mean(local_rho))
            print("Error bar:", error)
            print()



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