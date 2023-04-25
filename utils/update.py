#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9

import torch
from torch import nn
import numpy as np
import copy
import matplotlib.pyplot as plt
import os
import sys
from .transformer_utils import generate_square_subsequent_mask
sys.path.append("..")
from find_trace import *


# sys.path.append(os.getcwd())
# from network import generate_square_subsequent_mask

from generate_STL import generate_property, generate_property_test, get_robustness_score




def property_loss_simp(y_pred, property, loss_function):
    return torch.sum(loss_function(y_pred - property))



def property_loss_eventually(y_pred, property, loss_function, type):
    iterval = 2
    if type == "eventually-upper":
        diff_yp = y_pred - property
        unsqueezed_diff = diff_yp.view(diff_yp.shape[0], int(diff_yp.shape[1]//iterval), iterval)
        diff_min, ind = torch.min(loss_function(unsqueezed_diff), dim=2)
        return torch.sum(diff_min)
    elif type == "eventually-lower":
        diff_yp = property - y_pred
        unsqueezed_diff = diff_yp.view(diff_yp.shape[0], int(diff_yp.shape[1]//iterval), iterval)
        diff_min, ind = torch.min(loss_function(unsqueezed_diff), dim=2)
        return torch.sum(diff_min)



def property_loss(X, y_pred, property_by_station_day, loss_function, y, show):
    loss = torch.zeros(1).to('cuda')
    for ind, arr in enumerate(X):   # X shape: 64, 120, 3
        sensor = int(arr[-1,2].item())
        day = int(arr[-1,1].item())
        day = (day+1)%7
        loss += torch.sum(loss_function(y_pred[ind] - torch.tensor(property_by_station_day[sensor][day][0]).to('cuda')))
        loss += torch.sum(loss_function(torch.tensor(property_by_station_day[sensor][day][1]).to('cuda') - y_pred[ind]))
    
    if show:
        plt.plot(property_by_station_day[sensor][day][0])
        plt.plot(property_by_station_day[sensor][day][1])
        plt.plot(y[ind].detach().cpu())
        plt.plot(y_pred[ind].detach().cpu())
        plt.show()

    return loss




def repackage_hidden(h):
    """
    Wraps hidden states in new Tensors, to detach them from their history.
    """
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
    


def transformer_test(dataloader, net, args, loss_func):
    pass



def transformer_train(dataloader, net, args, loss_func, lr):
    bias_p, weight_p = [], []
    for name, p in net.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    
    optimizer = torch.optim.SGD([{'params': weight_p, 'weight_decay':0.0001}, {'params': bias_p, 'weight_decay':0}], lr=lr, momentum=0.5)
    local_eps = args.client_iter  # local update epochs
    src_mask = generate_square_subsequent_mask(dim1=24, dim2=120).cuda()
    tgt_mask = generate_square_subsequent_mask(dim1=24, dim2=24).cuda()

    epoch_loss = []
    for iter in range(local_eps):
        num_updates = 0
        for name, param in net.named_parameters():
            param.requires_grad = True 
        
        batch_loss = []
        for batch_idx, (X, y) in enumerate(dataloader): ## batch first=True
            net.train()
            optimizer.zero_grad()
            X = X.unsqueeze(2)
            output = net(src=X, tgt=X[:,-24:,:], src_mask=src_mask, tgt_mask=tgt_mask)
            loss = loss_func(output.view(-1, 24), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
            num_updates += 1
            batch_loss.append(loss.item())
            
            if num_updates == args.local_updates:
                break

        epoch_loss.append(sum(batch_loss)/len(batch_loss))
    
    return net, sum(epoch_loss)/len(epoch_loss)




# local updates for FedRep, FedPer, LG-FedAvg, FedAvg, FedProx
class LocalUpdate(object):
    """
    Federated learning updating class for a single agent.
    """
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.MSELoss()
        self.ldr_train = dataset["train_private"]
        self.ldr_val = dataset["val"]
        self.ldr_test = dataset["test"]
        self.idxs = idxs  # client index


    def train(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.001):
        if net.model_type == 'transformer':
            net, avg_ep_loss = transformer_train(self.ldr_train, net, self.args, self.loss_func, lr)
            return net.state_dict(), avg_ep_loss, self.idxs
        
        else:
            # get weights / bias parameter names
            bias_p, weight_p = [], []
            for name, p in net.named_parameters():
                if 'bias' in name:
                    bias_p += [p]
                else:
                    weight_p += [p]
            
            # use SGD as FedAvg
            optimizer = torch.optim.SGD([{'params': weight_p, 'weight_decay':0.0001}, {'params': bias_p, 'weight_decay':0}], lr=lr, momentum=0.5)
            local_eps = self.args.client_iter  # local update epochs
            
            epoch_loss = []
            hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()
        
            for iter in range(local_eps):   # for # total local ep
                num_updates = 0

                for name, param in net.named_parameters():
                    param.requires_grad = True 

                batch_loss = []
                for batch_idx, (X, y) in enumerate(self.ldr_train):
                    net.train()
                    optimizer.zero_grad()
                    hidden_1 = repackage_hidden(hidden_1)
                    hidden_2 = repackage_hidden(hidden_2)
                    output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)
                    loss = self.loss_func(output, y)
                    loss.backward()
                    optimizer.step()

                    num_updates += 1
                    batch_loss.append(loss.item())
                    
                    if num_updates == self.args.local_updates:
                        break

                epoch_loss.append(sum(batch_loss)/len(batch_loss))

            ## check that gt is within upper and lower property
            # output_p = output.detach().cpu().numpy()
            # target_p = y.cpu().numpy()
            # plt.plot(output_p[0], label='out')
            # plt.plot(target_p[0], label='gt')
            # plt.legend()
            # plt.show()
        
            return net.state_dict(), sum(epoch_loss)/len(epoch_loss), self.idxs



    def train_cluster(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.001):
        
        # get weights / bias parameter names
        bias_p, weight_p = [], []
        for name, p in net.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        
        # use SGD as FedAvg
        optimizer = torch.optim.SGD([{'params': weight_p, 'weight_decay':0.0001}, {'params': bias_p, 'weight_decay':0}], lr=lr, momentum=0.5)
        local_eps = self.args.client_iter  # local update epochs
        
        epoch_loss = []
        if net.model_type != 'transformer':
            hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()
            
            for iter in range(local_eps):   # for # total local ep
                num_updates = 0

                for name, param in net.named_parameters():
                    param.requires_grad = True 

                batch_loss = []
                for batch_idx, (X, y) in enumerate(self.ldr_train):
                    net.train()
                    optimizer.zero_grad()
                    hidden_1 = repackage_hidden(hidden_1)
                    hidden_2 = repackage_hidden(hidden_2)
                    output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)
                    loss = self.loss_func(output, y)
                    loss.backward()
                    optimizer.step()

                    num_updates += 1
                    batch_loss.append(loss.item())
                    
                    if num_updates == self.args.local_updates:
                        break

                epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        return net.state_dict(), sum(epoch_loss)/len(epoch_loss), self.idxs



    def test(self, net, w_glob_keys, dataset_test=None, ind=-1, idx=-1):
        epoch_loss = []
        num_updates = 0
        if net.model_type != 'transformer':
            hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()

            for name, param in net.named_parameters():
                param.requires_grad = False 

            batch_loss = []
            batch_y_test = []
            batch_pred_test = []
            for X, y in self.ldr_train:
                net.eval()
                hidden_1 = repackage_hidden(hidden_1)
                hidden_2 = repackage_hidden(hidden_2)
                output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)
                loss = self.loss_func(output, y)

                batch_y_test.append(y.detach().cpu().numpy())
                batch_pred_test.append(output.detach().cpu().numpy())
                
                net.zero_grad()
                num_updates += 1
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # for name, param in net.named_parameters():
        #     print(name, param.data)
        
        
        return net.state_dict(), sum(epoch_loss)/len(epoch_loss), self.idxs





def compute_cluster_id(cluster_models, client_dataset, args, idxs_users):
    cluster_loss = np.full((args.cluster, args.client), np.inf)
    
    # load cluster models 
    for cluster in range(args.cluster):
        for c in idxs_users:
            local = LocalUpdate(args=args, dataset=client_dataset[c], idxs=c)
            w_local, loss, idx = local.test(net=cluster_models[cluster] .to(args.device), idx=c, w_glob_keys=None)
            cluster_loss[cluster][c] = loss
    
    cluster_id = {}
    for cluster in range(args.cluster):
        cluster_id[cluster] = []
    
    client_lst = [c for c in idxs_users]
    i = 0
    while i < len(idxs_users):
        min_index = np.argwhere(cluster_loss == np.min(cluster_loss))
        if len(cluster_id[min_index[0][0]]) < 2 and min_index[0][1] in client_lst:
            cluster_id[min_index[0][0]].append(min_index[0][1])
            i += 1
            client_lst.remove(min_index[0][1])
        cluster_loss[min_index[0][0], min_index[0][1]] = np.inf

    return cluster_id




def cluster_id_property(cluster_models, client_dataset, args, idxs_users):
    cluster_loss = np.full((args.cluster, args.client), np.inf)
    
    # load cluster models 
    for cluster in range(args.cluster):
        for c in idxs_users:
            local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c)
            w_local, loss, cons_loss, idx = local.test(net=cluster_models[cluster] .to(args.device), idx=c, w_glob_keys=None)
            cluster_loss[cluster][c] = cons_loss
    
    cluster_id = {}
    for cluster in range(args.cluster):
        cluster_id[cluster] = []
    
    client_lst = [c for c in idxs_users]
    i = 0
    while i < len(idxs_users):
        min_index = np.argwhere(cluster_loss == np.min(cluster_loss))
        if len(cluster_id[min_index[0][0]]) < 2 and min_index[0][1] in client_lst:
            cluster_id[min_index[0][0]].append(min_index[0][1])
            i += 1
            client_lst.remove(min_index[0][1])
        cluster_loss[min_index[0][0], min_index[0][1]] = np.inf

    return cluster_id



def cluster_explore(net, w_glob_keys, lr, args, dataloaders, idxs):
    bias_p, weight_p = [], []
    for name, p in net.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]

    loss_func = nn.MSELoss()
    m = nn.ReLU()
    optimizer = torch.optim.SGD([{'params': weight_p, 'weight_decay':0.0001}, {'params': bias_p, 'weight_decay':0}], lr=lr, momentum=0.5)
   
    epoch_loss = []
    epoch_cons_loss = []

    if net.model_type != 'transformer':
        hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()

        for name, param in net.named_parameters():
            if name in w_glob_keys:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        for iter in range(args.cluster_fine_tune_iter):
            num_updates = 0
            batch_loss = []
            batch_cons_loss = []

            for batch_idx, (X, y) in enumerate(dataloaders):
                net.train()
                optimizer.zero_grad()
                hidden_1 = repackage_hidden(hidden_1)
                hidden_2 = repackage_hidden(hidden_2)
                output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)
                
                pred_loss = loss_func(output, y)
                
                if args.property_type == 'corr':
                    property_mined = generate_property(X, property_type = args.property_type)
                    cons_loss = loss_func(torch.mean(output[:,:,0], dim=0)-torch.mean(output[:,:,1], dim=0), property_mined[0])
                elif args.property_type == 'constraint':
                    property_upper = generate_property_test(X, property_type = "upper")
                    property_lower = generate_property_test(X, property_type = "lower")
                    cons_loss = property_loss_simp(output, property_upper, m) + property_loss_simp(property_lower, output, m)
                elif args.property_type == 'eventually':
                    property_upper = generate_property_test(X, property_type = "eventually-upper")
                    property_lower = generate_property_test(X, property_type = "eventually-lower")
                    cons_loss = property_loss_eventually(output, property_upper, m, "eventually-upper") + property_loss_eventually(output, property_lower, m, "eventually-lower")
                else:
                    raise NotImplementedError

                while abs(cons_loss) > pred_loss:
                    cons_loss = cons_loss/10
                
                loss = pred_loss + cons_loss
                loss.backward()
                optimizer.step()

                num_updates += 1
                batch_loss.append(pred_loss.item())
                batch_cons_loss.append(cons_loss)
                
                if num_updates == args.local_updates: break
                
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_cons_loss.append(sum(batch_cons_loss)/len(batch_cons_loss))
    
    return net.state_dict(), sum(epoch_loss)/len(epoch_loss)



class LocalUpdateProp(object):
    """
    Federated learning updating class for a single agent with property mining.
    """
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.MSELoss(reduction='mean')
        self.m = nn.ReLU()
        self.ldr_train = dataset["train_private"]
        self.ldr_val = dataset["val"]
        self.ldr_test = dataset["test"]
        self.idxs = idxs  # client index


    def train(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.001):
        # get weights / bias parameter names
        bias_p, weight_p = [], []
        for name, p in net.named_parameters():
            if 'bias' in name: 
                bias_p += [p]
            else: 
                weight_p += [p]
       
        optimizer = torch.optim.SGD([{'params': weight_p, 'weight_decay':0.0001}, {'params': bias_p, 'weight_decay':0}], lr=lr, momentum=0.5)
        local_eps = self.args.client_iter   # local update epochs
        
        epoch_loss = []
        epoch_cons_loss = []

        if net.model_type != 'transformer':
            hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()
            
            for iter in range(local_eps):   # for # total local ep
                num_updates = 0

                for name, param in net.named_parameters():
                    param.requires_grad = True 

                batch_loss = []
                batch_cons_loss = []
                for batch_idx, (X, y) in enumerate(self.ldr_train):
                    net.train()
                    optimizer.zero_grad()
                    hidden_1 = repackage_hidden(hidden_1)
                    hidden_2 = repackage_hidden(hidden_2)
                    output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)
                    pred_loss = self.loss_func(output, y)
                    
                    if self.args.property_type == 'corr':
                        property_mined = generate_property(X, property_type = self.args.property_type)
                        cons_loss = self.loss_func(torch.mean(output[:,:,0], dim=0)-torch.mean(output[:,:,1], dim=0), property_mined[0])
                    elif self.args.property_type == 'constraint':
                        property_upper, stl_lib_upper = generate_property_test(X, property_type = "upper")
                        corrected_trace_upper = convert_best_trace(stl_lib_upper, output)
                        property_lower, stl_lib_lower = generate_property_test(X, property_type = "lower")
                        corrected_trace_lower = convert_best_trace(stl_lib_lower, output)
                        # cons_loss = property_loss_simp(output, property_upper, self.m) + property_loss_simp(property_lower, output, self.m)
                        ## check that gt is within upper and lower property
                        # output_p = output.detach().cpu().numpy()
                        # target_p = y.detach().cpu().numpy()
                        # print(corrected_trace_upper[0].cpu().numpy())
                        # print(corrected_trace_lower[0].cpu().numpy())
                        # plt.plot(output_p[0], label='out')
                        # plt.plot(target_p[0], label='gt')
                        # plt.plot(corrected_trace_upper[0].detach().cpu().numpy(), label='up')
                        # plt.plot(corrected_trace_lower[0].detach().cpu().numpy(), label='low')
                        # plt.legend()
                        # plt.show()
                        cons_loss = self.loss_func(output, corrected_trace_upper) + self.loss_func(output, corrected_trace_lower)
                    
                    elif self.args.property_type == 'eventually':
                        property_upper, stl_lib = generate_property_test(X, property_type = "eventually-upper")
                        property_lower, stl_lib = generate_property_test(X, property_type = "eventually-lower")
                        cons_loss = property_loss_eventually(output, property_upper, self.m, "eventually-upper") + property_loss_eventually(output, property_lower, self.m, "eventually-lower")
                    else:
                        raise NotImplementedError
                    


                    # while abs(cons_loss) > pred_loss:
                    #     cons_loss = cons_loss/10
                    
                    loss = pred_loss + cons_loss
                    loss.backward()
                    optimizer.step()

                    num_updates += 1
                    batch_loss.append(pred_loss.item())
                    batch_cons_loss.append(cons_loss.item())
                    
                    if num_updates == self.args.local_updates: 
                        break

                epoch_loss.append(sum(batch_loss)/len(batch_loss))
                epoch_cons_loss.append(sum(batch_cons_loss)/len(batch_cons_loss))
            
            ## check that gt is within upper and lower property
            # output_p = output.detach().cpu().numpy()
            # target_p = y.cpu().numpy()
            # plt.plot(output_p[0], label='out')
            # plt.plot(target_p[0], label='gt')
            # plt.plot(property_upper[0,:24].cpu().numpy(), label='up')
            # plt.plot(property_lower[0,:24].cpu().numpy(), label='low')
            # plt.legend()
            # plt.show()
        
        return net.state_dict(), sum(epoch_loss)/len(epoch_loss), self.idxs



    def test(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.001, rho=False):
        m = nn.ReLU()
        epoch_rho = []
        epoch_loss = []
        epoch_cons_loss = []
        num_updates = 0

        if net.model_type != 'transformer':
            hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()

            for name, param in net.named_parameters():
                param.requires_grad = False

            batch_rho = []
            batch_loss = []
            batch_cons_loss = []
            for X, y in self.ldr_train: ## TODO: fix val and test
                net.eval()
                hidden_1 = repackage_hidden(hidden_1)
                hidden_2 = repackage_hidden(hidden_2)
                output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)
                pred_loss = self.loss_func(output, y)

                if self.args.property_type == 'corr':
                    property_mined = generate_property(X, property_type = self.args.property_type)
                    cons_loss = self.loss_func(torch.mean(output[:,:,0], dim=0)-torch.mean(output[:,:,1], dim=0), property_mined[0])
                elif self.args.property_type == 'constraint':
                    property_upper, _ = generate_property_test(X, property_type = "upper")
                    property_lower, _ = generate_property_test(X, property_type = "lower")
                    # cons_loss = property_loss_simp(output, property_upper, self.m) + property_loss_simp(property_lower, output, self.m)
                    # output_p = output.detach().cpu().numpy()
                    # target_p = y.cpu().numpy()
                    # print(property_upper.cpu().numpy())
                    # print(property_lower.cpu().numpy())
                    # plt.plot(output_p[0])
                    # plt.plot(target_p[0])
                    # plt.plot(property_upper[0, :24].cpu().numpy())
                    # plt.plot(property_upper[0, :24].cpu().numpy())
                    # plt.show()
                    cons_loss = self.loss_func(output, property_upper[:,:24]) + self.loss_func(output, property_lower[:,:24])
                elif self.args.property_type == 'eventually':
                    property_upper, _ = generate_property_test(X, property_type = "eventually-upper")
                    property_lower, _ = generate_property_test(X, property_type = "eventually-lower")
                    cons_loss = property_loss_eventually(output, property_upper, self.m, "eventually-upper") + property_loss_eventually(output, property_lower, self.m, "eventually-lower")
                else:
                    raise NotImplementedError

                # while abs(cons_loss) > pred_loss:
                #     cons_loss = cons_loss/10

                batch_loss.append(pred_loss.item())
                batch_cons_loss.append(cons_loss)

                if rho==True:
                    # used version
                    if self.args.property_type == 'corr':
                        diff_out = torch.mean(output[:,:-1,0], dim=0)-torch.mean(output[:,:-1,1], dim=0)
                        diff_mine = property_mined[0,1:]
                        batch_rho.append( 1-torch.count_nonzero(m(diff_mine - diff_out))/len(diff_out) )
                    elif self.args.property_type == 'constraint':
                        batch_rho.append( 1-torch.count_nonzero(m(property_lower - output))/len(property_lower)/property_lower.shape[1] )
                        batch_rho.append( 1-torch.count_nonzero(m(output - property_upper))/len(property_upper)/property_upper.shape[1] )
                    elif self.args.property_type == 'eventually':
                        iterval = 2
                        diff_yp = output - property_upper
                        unsqueezed_diff = diff_yp.view(diff_yp.shape[0], int(diff_yp.shape[1]//iterval), iterval)
                        diff_min, ind = torch.min(m(unsqueezed_diff), dim=2)
                        batch_rho.append( 1-torch.count_nonzero(diff_min) / len(diff_min) / diff_min.shape[1] )
                        diff_yp = property_lower - output
                        unsqueezed_diff = diff_yp.view(diff_yp.shape[0], int(diff_yp.shape[1]//iterval), iterval)
                        diff_min, ind = torch.min(m(unsqueezed_diff), dim=2)
                        batch_rho.append( 1-torch.count_nonzero(diff_min) / len(diff_min) / diff_min.shape[1] )
                    else:
                        raise NotImplementedError

                num_updates += 1
                # if num_updates >= 25: break

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_cons_loss.append(sum(batch_cons_loss)/len(batch_cons_loss))
            if rho==True:
                epoch_rho.append(sum(batch_rho)/len(batch_rho))
            
        if rho==True:
            return net.state_dict(), sum(epoch_loss)/len(epoch_loss), sum(epoch_cons_loss)/len(epoch_cons_loss), self.idxs, sum(epoch_rho)/len(epoch_rho)
        else:
            return net.state_dict(), sum(epoch_loss)/len(epoch_loss), sum(epoch_cons_loss)/len(epoch_cons_loss), self.idxs


    def train_cluster(self, net, w_glob_keys=['lstm_1.weight_ih', 'lstm_1.weight_hh', 'lstm_1.bias_ih', 'lstm_1.bias_hh'], last=False, dataset_test=None, ind=-1, idx=-1, lr=0.001):
        """
        cluster training with fine-tuning with property
        """
        bias_p, weight_p = [], []
        for name, p in net.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]

        m = nn.ReLU()
        optimizer = torch.optim.SGD([{'params': weight_p, 'weight_decay':0.0001}, {'params': bias_p, 'weight_decay':0}], lr=lr, momentum=0.5)
        local_eps = self.args.client_iter               # 15
        head_eps = local_eps-self.args.fine_tune_iter   # 15-5 = 10
        
        epoch_loss = []
        epoch_cons_loss = []
        if net.model_type != 'transformer':
            hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()
            
            for iter in range(local_eps):   # for # total local ep
                done = False
                num_updates = 0

                if last:
                    for name, param in net.named_parameters():
                        param.requires_grad = True

                elif iter < head_eps:
                    for name, param in net.named_parameters():
                        if name in w_glob_keys:
                            param.requires_grad = True
                        else:
                            param.requires_grad = True

                elif iter >= head_eps:
                    for name, param in net.named_parameters():
                        if name in w_glob_keys:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True

                batch_loss = []
                batch_cons_loss = []
                for batch_idx, (X, y) in enumerate(self.ldr_train):
                    net.train()
                    optimizer.zero_grad()
                    hidden_1 = repackage_hidden(hidden_1)
                    hidden_2 = repackage_hidden(hidden_2)
                    output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)
                    
                    pred_loss = self.loss_func(output, y)
                    
                    if iter >= head_eps:    # calc property loss for heads 
                        if self.args.property_type == 'corr':
                            property_mined = generate_property(X, property_type = self.args.property_type)
                            cons_loss = self.loss_func(torch.mean(output[:,:,0], dim=0)-torch.mean(output[:,:,1], dim=0), property_mined[0])
                        elif self.args.property_type == 'constraint':
                            property_upper = generate_property_test(X, property_type = "upper")
                            property_lower = generate_property_test(X, property_type = "lower")
                            cons_loss = property_loss_simp(output, property_upper, self.m) + property_loss_simp(property_lower, output, self.m)
                        elif self.args.property_type == 'eventually':
                            property_upper = generate_property_test(X, property_type = "eventually-upper")
                            property_lower = generate_property_test(X, property_type = "eventually-lower")
                            cons_loss = property_loss_eventually(output, property_upper, self.m, "eventually-upper") + property_loss_eventually(output, property_lower, self.m, "eventually-lower")
                        else:
                            raise NotImplementedError
                        
                        while abs(cons_loss) > pred_loss:
                            cons_loss = cons_loss/10
                        loss = pred_loss + cons_loss
                    else:
                        loss = pred_loss

                    loss.backward()
                    optimizer.step()

                    num_updates += 1
                    batch_loss.append(pred_loss.item())
                    if iter >= head_eps:
                        batch_cons_loss.append(cons_loss)
                    
                    if num_updates == self.args.local_updates:
                        done = True
                        break

                epoch_loss.append(sum(batch_loss)/len(batch_loss))
                if iter >= head_eps:
                    epoch_cons_loss.append(sum(batch_cons_loss)/len(batch_cons_loss))
                
        return net.state_dict(), sum(epoch_loss)/len(epoch_loss), self.idxs