import torch
import numpy as np
from IoTData import SequenceDataset
from torch.utils.data import DataLoader
import os



def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")



def to_device(data,device):
    if isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking=True)



def save_model(path, model, model_name, epoch):
    if not os.path.isdir(path):
        os.makedirs(path)
    save_prefix = os.path.join(path, model_name)
    save_path = '{}_epoch_{}.pt'.format(save_prefix, epoch)
    print("save all model to {}".format(save_path))
    output = open(save_path, mode="wb")
    torch.save(model.state_dict(), output)
    output.close() 



def get_client_dataset(client_id, dataset_name):
    dataset_array = {}
    if dataset_name == 'fhwa':
        dataset_path = "/hdd/FHWA_dataset/torch_dataset/"
    elif dataset_name == 'sumo':
        dataset_path = "/hdd/SUMO_dataset/learn_dataset/"
    for fold in ["train", "test", "val"]:
        x_file = dataset_path+fold+"_"+str(client_id)+"_x.npy"
        y_file = dataset_path+fold+"_"+str(client_id)+"_y.npy"
        dataset_array[fold+"_x"] = np.load(x_file, allow_pickle=True)
        dataset_array[fold+"_y"] = np.load(y_file, allow_pickle=True)
    train_dataset = SequenceDataset(dataset_array["train_x"], dataset_array["train_y"])
    val_dataset = SequenceDataset(dataset_array["val_x"], dataset_array["val_y"])
    test_dataset = SequenceDataset(dataset_array["test_x"], dataset_array["test_y"])
    dataset_len = [len(train_dataset), len(val_dataset), len(test_dataset)]
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=64, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=64, drop_last=True)

    return train_loader, val_loader, test_loader, dataset_len



def get_shared_dataset(client_id, dataset_name):
    dataset_array = {}
    if dataset_name == 'fhwa':
        dataset_path = "/hdd/FHWA_dataset/torch_dataset/"
    elif dataset_name == 'sumo':
        dataset_path = "/hdd/SUMO_dataset/learn_dataset/"
    for fold in ["train", "test", "val"]:
        x_file = dataset_path+fold+"_"+str(client_id)+"_x.npy"
        y_file = dataset_path+fold+"_"+str(client_id)+"_y.npy"
        dataset_array[fold+"_x"] = np.load(x_file, allow_pickle=True)
        dataset_array[fold+"_y"] = np.load(y_file, allow_pickle=True)
    
    public_len = int(0.2*len(dataset_array["train_x"]))
    train_dataset = SequenceDataset(dataset_array["train_x"][public_len:], dataset_array["train_y"][public_len:])
    val_dataset = SequenceDataset(dataset_array["val_x"], dataset_array["val_y"])
    test_dataset = SequenceDataset(dataset_array["test_x"], dataset_array["test_y"])

    dataset_len = [len(train_dataset), len(val_dataset), len(test_dataset)]
    train_loader_private = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=64, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=64, drop_last=True)

    return train_loader_private, [dataset_array["train_x"][:public_len], dataset_array["train_y"][:public_len]], val_loader, test_loader, dataset_len