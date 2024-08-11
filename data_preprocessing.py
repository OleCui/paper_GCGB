import os
import dgl
import torch
import random
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from parse_args import args
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import HeteroData

def get_data(original_data_path):
    d_data = dict()

    drf = pd.read_csv(os.path.join(original_data_path, 'DrugFingerprint.csv')).iloc[:, 1:].to_numpy()
    drg = pd.read_csv(os.path.join(original_data_path, 'DrugGIP.csv')).iloc[:, 1:].to_numpy()

    dip = pd.read_csv(os.path.join(original_data_path, 'DiseasePS.csv')).iloc[:, 1:].to_numpy()
    dig = pd.read_csv(os.path.join(original_data_path, 'DiseaseGIP.csv')).iloc[:, 1:].to_numpy()

    d_data['drug_number'] = int(drf.shape[0])
    d_data['disease_number'] = int(dig.shape[0])

    d_data['drf'] = drf
    d_data['drg'] = drg
    d_data['dip'] = dip
    d_data['dig'] = dig

    d_data['drdi'] = pd.read_csv(os.path.join(original_data_path, 'DrugDiseaseAssociationNumber.csv'), dtype=int).to_numpy()
    
    d_data['drpr'] = pd.read_csv(os.path.join(original_data_path, 'DrugProteinAssociationNumber.csv'), dtype=int).to_numpy()
    d_data['dipr'] = pd.read_csv(os.path.join(original_data_path, 'ProteinDiseaseAssociationNumber.csv'), dtype=int).to_numpy()

    d_data['drugfeature'] = pd.read_csv(os.path.join(original_data_path,'Drug_mol2vec.csv'), header=None).iloc[:, 1:].to_numpy()
    d_data['diseasefeature'] = pd.read_csv(os.path.join(original_data_path,'DiseaseFeature.csv'), header=None).iloc[:, 1:].to_numpy()
    d_data['proteinfeature'] = pd.read_csv(os.path.join(original_data_path,'Protein_ESM.csv'), header=None).iloc[:, 1:].to_numpy()

    d_data['drugfeature'] = torch.FloatTensor(d_data['drugfeature'])
    d_data['diseasefeature'] = torch.FloatTensor(d_data['diseasefeature'])
    d_data['proteinfeature'] = torch.FloatTensor(d_data['proteinfeature'])

    d_data['protein_number']= d_data['proteinfeature'].shape[0]

    return d_data

def get_adj(edges, size):
    edges_tensor = torch.LongTensor(edges).t()
    values = torch.ones(len(edges))
    adj = torch.sparse.LongTensor(edges_tensor, values, size).to_dense().long() 

    return adj

def generate_drug_disease_training_samples(d_data):
    drdi_matrix = get_adj(d_data['drdi'], (d_data['drug_number'], d_data['disease_number']))

    l_one_index = []
    l_zero_index = []
    
    for i in range(drdi_matrix.shape[0]):
        for j in range(drdi_matrix.shape[1]):
            if drdi_matrix[i][j] >= 1:
                l_one_index.append([i, j])
            else:
                l_zero_index.append([i, j])
    
    random.seed(args.seed)
    random.shuffle(l_one_index)
    random.shuffle(l_zero_index)
    l_zero_index = l_zero_index[:int(args.negative_rate * len(l_one_index))]
    index = np.array(l_one_index + l_zero_index, dtype=int)
    label = np.array([1] * len(l_one_index) + [0] * len(l_zero_index), dtype=int) 
    samples = np.concatenate((index, np.expand_dims(label, axis=1)), axis=1)

    drdi_p = samples[samples[:, 2] == 1, :]
    drdi_n = samples[samples[:, 2] == 0, :]
    
    drs_mean = (d_data['drf'] + d_data['drg']) / 2
    dis_mean = (d_data['dip'] + d_data['dig']) / 2

    drs = np.where(d_data['drf'] == 0, d_data['drg'], drs_mean)
    dis = np.where(d_data['dip'] == 0, d_data['dig'], dis_mean)

    d_data['dr_sim'] = drs
    d_data['di_sim'] = dis
    d_data['all_drdi'] = samples[:, :2]
    d_data['all_drdi_p'] = drdi_p
    d_data['all_drdi_n'] = drdi_n
    d_data['all_label'] = label

def K_fold(d_data, args, output_data_path):
    skf = StratifiedKFold(n_splits=args.K_fold, random_state=None, shuffle=False)
    X = d_data['all_drdi']
    Y = d_data['all_label']
    X_train_all, X_test_all, Y_train_all, Y_test_all = [], [], [], []

    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        Y_train = np.expand_dims(Y_train, axis=1).astype('float64')
        Y_test = np.expand_dims(Y_test, axis=1).astype('float64')
        X_train_all.append(X_train)
        X_test_all.append(X_test)
        Y_train_all.append(Y_train)
        Y_test_all.append(Y_test)
    
    for i in range(args.K_fold):
        data_path_i = os.path.join(output_data_path, 'seed_{}_negRate_{}_KFold_{}/{}'.format(args.seed, args.negative_rate, args.K_fold, i))
        if not os.path.exists(data_path_i):
            os.makedirs(data_path_i)

            X_train = pd.DataFrame(data=np.concatenate((X_train_all[i], Y_train_all[i]), axis=1), columns=['drug', 'disease', 'label'])
            X_train.to_csv(os.path.join(data_path_i, 'data_train.csv'))

            X_test = pd.DataFrame(data=np.concatenate((X_test_all[i], Y_test_all[i]), axis=1), columns=['drug', 'disease', 'label'])
            X_test.to_csv(os.path.join(data_path_i, 'data_test.csv'))
    
    d_data['X_train'] = X_train_all
    d_data['X_test'] = X_test_all
    d_data['Y_train'] = Y_train_all
    d_data['Y_test'] = Y_test_all

def KNN_matrix(matrix, k, isBool = True):
    num = matrix.shape[0]
    knn_graph = np.zeros(matrix.shape, dtype=int)
    idx_sort = np.argsort(-(matrix - np.eye(num)), axis=1)

    for i in range(num):
        if isBool:
            knn_graph[i, idx_sort[i, :k]] = 1
            knn_graph[idx_sort[i, :k], i] = 1
        else:
            knn_graph[i, idx_sort[i, :k]] = matrix[i, idx_sort[i, :k]]
            knn_graph[idx_sort[i, :k], i] = matrix[idx_sort[i, :k], i]

        knn_graph[i, i] = 1
    return knn_graph

def process_similarity_graph(d_data, args):
    drdr_matrix_bool = KNN_matrix(d_data['dr_sim'], args.KNN_neighbor, isBool = True)
    didi_matrix_bool = KNN_matrix(d_data['di_sim'], args.KNN_neighbor, isBool = True)

    d_data["drdr_matrix"] = drdr_matrix_bool
    d_data["didi_matrix"] = didi_matrix_bool

    drdr_matrix = KNN_matrix(d_data['dr_sim'], args.KNN_neighbor, isBool = False) 
    didi_matrix = KNN_matrix(d_data['di_sim'], args.KNN_neighbor, isBool = False)

    drdr_nx = nx.from_numpy_array(drdr_matrix)
    didi_nx = nx.from_numpy_array(didi_matrix)

    drdr_graph = dgl.from_networkx(drdr_nx)
    didi_graph = dgl.from_networkx(didi_nx)

    drdr_graph.ndata['sim_feature'] = torch.tensor(d_data['dr_sim'])
    didi_graph.ndata['sim_feature'] = torch.tensor(d_data['di_sim'])

    return drdr_graph, didi_graph

def process_data():
    original_data_path = os.path.join(os.getcwd(), "Datasets/{}".format(args.dataset))

    output_data_path = os.path.join(os.getcwd(), "Outputs/{}".format(args.dataset))

    if not os.path.exists(output_data_path):
        os.makedirs(output_data_path)
    
    d_data = get_data(original_data_path)

    generate_drug_disease_training_samples(d_data)

    K_fold(d_data, args, output_data_path)

    drdr_graph, didi_graph = process_similarity_graph(d_data, args)

    return d_data, drdr_graph, didi_graph

def dgl_heterograph(d_data, drdi):
    l_drdi, l_drpr, l_dipr = [], [], []
    l_didr, l_prdr, l_prdi = [], [], []

    for i in range(drdi.shape[0]):
        l_drdi.append(drdi[i])
        l_didr.append(drdi[i][::-1])

    for i in range(d_data['drpr'].shape[0]):
        l_drpr.append(d_data['drpr'][i])
        l_prdr.append(d_data['drpr'][i][::-1])

    for i in range(d_data['dipr'].shape[0]):
        l_dipr.append(d_data['dipr'][i])
        l_prdi.append(d_data['dipr'][i][::-1])

    node_dict = {
        'drug': d_data['drug_number'],
        'disease': d_data['disease_number'],
        'protein': d_data['protein_number']
    }

    heterograph_dict = {
        ('drug', 'association', 'disease'): (l_drdi),
        ('drug', 'association', 'protein'): (l_drpr),
        ('disease', 'association', 'protein'): (l_dipr),

        ('disease', 'association', 'drug'): (l_didr),
        ('protein', 'association', 'drug'): (l_prdr),
        ('protein', 'association', 'disease'): (l_prdi)

    }

    heterograph = dgl.heterograph(heterograph_dict, num_nodes_dict=node_dict)

    return heterograph