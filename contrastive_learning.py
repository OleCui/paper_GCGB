import torch
import numpy as np
from parse_args import args
import torch.nn.functional as F

EPSILON = float(np.finfo(float).eps)

def safe_log(x):
    return torch.log(x + EPSILON)

def get_pos_neg_indice(drug_simlilarity_matrix, disease_simlilarity_matrix):
    drug_num = drug_simlilarity_matrix.shape[0]
    disease_num = disease_simlilarity_matrix.shape[0]

    drug_pos_indice = drug_simlilarity_matrix - np.eye(drug_num)
    disease_pos_indice = disease_simlilarity_matrix - np.eye(disease_num)

    drug_pos_indice = torch.from_numpy(drug_pos_indice).long().cuda()
    disease_pos_indice = torch.from_numpy(disease_pos_indice).long().cuda()

    drug_neg_indice = torch.from_numpy(drug_simlilarity_matrix).long().cuda()
    drug_neg_indice = (drug_neg_indice == 0).long()

    disease_neg_indice = torch.from_numpy(disease_simlilarity_matrix).long().cuda()
    disease_neg_indice = (disease_neg_indice == 0).long()

    return drug_pos_indice, drug_neg_indice, disease_pos_indice, disease_neg_indice

def similarity_contrastive(drug_simlilarity_matrix, disease_simlilarity_matrix, drug_feature, disease_feature):
    drug_pos_indice, drug_neg_indice, disease_pos_indice, disease_neg_indice = get_pos_neg_indice(drug_simlilarity_matrix, disease_simlilarity_matrix)

    drug_feature = F.normalize(drug_feature, p=2, dim = 1)
    disease_feature = F.normalize(disease_feature, p=2, dim = 1)

    drug_feature_reshape = drug_feature.unsqueeze(dim = 1)
    disease_feature_reshape = disease_feature.unsqueeze(dim = 1)

    drug_score = torch.matmul(drug_feature_reshape, drug_feature.transpose(1, 0)).squeeze(1)
    disease_score = torch.matmul(disease_feature_reshape, disease_feature.transpose(1, 0)).squeeze(1)
    
    drug_pos_score = drug_score * drug_pos_indice
    drug_neg_score = drug_score * drug_neg_indice

    disease_pos_score = disease_score * disease_pos_indice
    disease_neg_score = disease_score * disease_neg_indice

    drug_pos_score = torch.exp(drug_pos_score / args.intra_ssl_temperature).sum(dim = 1)
    drug_neg_score = torch.exp(drug_neg_score / args.intra_ssl_temperature).sum(dim = 1)

    disease_pos_score = torch.exp(disease_pos_score / args.intra_ssl_temperature).sum(dim = 1)
    disease_neg_score = torch.exp(disease_neg_score / args.intra_ssl_temperature).sum(dim = 1)

    drug_contrastive = -torch.sum(safe_log(drug_pos_score / drug_neg_score))
    disease_contrastive = -torch.sum(safe_log(disease_pos_score / disease_neg_score))

    return drug_contrastive + disease_contrastive

def inter_contrastive(drug_simlilarity_matrix, disease_simlilarity_matrix, drug_feature1, disease_feature1, drug_feature2, disease_feature2):
    _, drug_neg_indice, _, disease_neg_indice = get_pos_neg_indice(drug_simlilarity_matrix, disease_simlilarity_matrix)

    drug_feature1 = F.normalize(drug_feature1, p=2, dim = 1)
    disease_feature1 = F.normalize(disease_feature1, p=2, dim = 1)
    drug_feature2 = F.normalize(drug_feature2, p=2, dim = 1)
    disease_feature2 = F.normalize(disease_feature2, p=2, dim = 1)

    drug_pos_score = torch.multiply(drug_feature1, drug_feature2).sum(dim=1)
    drug_neg_score = torch.matmul(drug_feature1, drug_feature2.t())

    disease_pos_score = torch.multiply(disease_feature1, disease_feature2).sum(dim=1)
    disease_neg_score = torch.matmul(disease_feature1, disease_feature2.t())
    
    drug_neg_score = drug_neg_indice * drug_neg_score
    disease_neg_score = disease_neg_indice * disease_neg_score

    drug_pos_score = torch.exp(drug_pos_score / args.inter_ssl_temperature)
    drug_neg_score = torch.exp(drug_neg_score / args.inter_ssl_temperature).sum(dim=1)

    disease_pos_score = torch.exp(disease_pos_score / args.inter_ssl_temperature)
    disease_neg_score = torch.exp(disease_neg_score / args.inter_ssl_temperature).sum(dim=1)

    drug_ssl_loss = -torch.log(drug_pos_score / drug_neg_score).sum()
    disease_ssl_loss = -torch.log(disease_pos_score / disease_neg_score).sum()

    return drug_ssl_loss + disease_ssl_loss