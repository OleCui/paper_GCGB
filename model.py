import math
import torch
import torch.nn as nn
import dgl.nn.pytorch
from parse_args import args
import torch.nn.functional as F
from graph_transformer import GraphTransformer

device = torch.device('cuda')

def multiple_operator(a, b):
    return a * b

def rotate_operator(a, b):
    a_re, a_im = a.chunk(2, dim=-1)
    b_re, b_im = b.chunk(2, dim=-1)
    message_re = a_re * b_re - a_im * b_im
    message_im = a_re * b_im + a_im * b_re
    message = torch.cat([message_re, message_im], dim=-1)
    return message

class Multi_Head_Self_ATT(nn.Module):

    def __init__(self, head_num, in_channels, out_channels):
        super(Multi_Head_Self_ATT, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.head_num = head_num

        self.k_lin = nn.Linear(self.in_channels, self.out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_lin.weight)
        nn.init.constant_(self.k_lin.bias, 0.0)

    def forward(self, x1, x2):
        x = torch.stack([x1, x2], dim = 1)
        B, M, _ = x.shape
        H, D = self.head_num, self.out_channels // self.head_num

        q = torch.mean(x, dim = 1).view(B, 1, H, D)
        k = self.k_lin(x).view(x.shape[:-1] + (H, D))
        v = x.view(x.shape[:-1] + (H, D))

        q = q.permute(0,2,1,3)
        k = k.permute(0,2,3,1)
        v = v.permute(0,2,1,3)

        alpha = F.softmax((q @ k / math.sqrt(q.size(-1))), dim=-1)

        o = alpha @ v 

        output = o.permute(0,2,1,3).reshape((B, H * D))

        return output

class Attention(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Attention, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.W1 = nn.Linear(self.in_channels, self.out_channels)
        self.W2 = nn.Linear(self.out_channels, 1)
        self.reset_parameters()


    def forward(self, tensor_a, tensor_b):
        feature = torch.stack([tensor_a, tensor_b], dim = 1)

        tensor = torch.tanh(self.W1(feature))
        tensor = self.W2(tensor).squeeze(2) 
        attention = F.softmax(tensor, dim=1).unsqueeze(1)

        output = torch.matmul(attention, feature).squeeze(1)

        return output

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)
        nn.init.constant_(self.W1.bias, 0.0)
        nn.init.constant_(self.W2.bias, 0.0)

class MyModel(nn.Module):
    def __init__(self, meta_g, drug_number, disease_number):
        super(MyModel, self).__init__()
        self.drug_number = drug_number
        self.meta_g = meta_g
        self.disease_number = disease_number

        self.drug_linear = nn.Linear(300, args.hgt_out_dim) 
        self.protein_linear = nn.Linear(320, args.hgt_out_dim)
        self.disease_linear = nn.Linear(64, args.hgt_out_dim)

        self.gt_drug = GraphTransformer(device, args.gt_layer, self.drug_number, args.gt_out_dim, args.gt_out_dim, args.gt_head, args.dropout)
        self.gt_disease = GraphTransformer(device, args.gt_layer, self.disease_number, args.gt_out_dim, args.gt_out_dim, args.gt_head, args.dropout)

        self.hgt_dgl = dgl.nn.pytorch.conv.HGTConv(args.hgt_out_dim, int(args.hgt_out_dim/args.hgt_head), args.hgt_head, len(self.meta_g.nodes()), len(self.meta_g.edges()), args.dropout)

        self.hgt = nn.ModuleList()
        for _ in range(args.hgt_layer):
            self.hgt.append(self.hgt_dgl)

        self.drug_fusion_atts = nn.ModuleList()
        self.disease_fusion_atts = nn.ModuleList()

        for _ in range(args.hgt_layer):
            att = Multi_Head_Self_ATT(4, args.hgt_out_dim, args.hgt_out_dim)
            self.drug_fusion_atts.append(att)
        
        for _ in range(args.hgt_layer):
            att = Multi_Head_Self_ATT(4, args.hgt_out_dim, args.hgt_out_dim)
            self.disease_fusion_atts.append(att)
        
        self.mlp = nn.Sequential(
        nn.Linear(args.gt_out_dim * 4, 1024),
        nn.ReLU(),
        nn.Dropout(args.dropout),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Dropout(args.dropout),
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Dropout(args.dropout),
        nn.Linear(256, 2))
        

    def forward(self, drdr_graph, didi_graph, drdipr_graph, drug_feature, disease_feature, protein_feature, sample):
        l_dr_sim = self.gt_drug(drdr_graph)
        l_di_sim = self.gt_disease(didi_graph)

        drug_feature = self.drug_linear(drug_feature)
        protein_feature = self.protein_linear(protein_feature)
        disease_feature = self.disease_linear(disease_feature)

        feature_dict = {
            'drug': drug_feature,
            'disease': disease_feature,
            'protein': protein_feature
        }

        drdipr_graph.ndata['h'] = feature_dict
        g = dgl.to_homogeneous(drdipr_graph, ndata='h')

        feature = torch.cat((drug_feature, disease_feature, protein_feature), dim=0)

        l_dr_ass = []
        l_di_ass = []
        l_dr_fusion = []
        l_di_fusion = []

        for layer in self.hgt:
            hgt_out = layer(g, feature, g.ndata['_TYPE'], g.edata['_TYPE'], presorted=True)
            feature = hgt_out
            l_dr_ass.append(hgt_out[:self.drug_number, :])
            l_di_ass.append(hgt_out[self.drug_number:self.disease_number + self.drug_number, :])

        
        for i, (dr_sim, dr_ass) in enumerate(zip(l_dr_sim, l_dr_ass)):
            dr_fusion_i = self.drug_fusion_atts[i](dr_sim, dr_ass)
            l_dr_fusion.append(dr_fusion_i)

        for i, (di_sim, di_ass) in enumerate(zip(l_di_sim, l_di_ass)):
            di_fusion_i = self.disease_fusion_atts[i](di_sim, di_ass)
            l_di_fusion.append(di_fusion_i)

        dr_fusion_final = torch.mean(torch.stack(l_dr_fusion, dim = 1), dim = 1)
        di_fusion_final = torch.mean(torch.stack(l_di_fusion, dim = 1), dim = 1)

        dr_sim_final = torch.mean(torch.stack(l_dr_sim, dim = 1), dim = 1)
        di_sim_final = torch.mean(torch.stack(l_di_sim, dim = 1), dim = 1)

        dr_ass_final = torch.mean(torch.stack(l_dr_ass, dim = 1), dim = 1)
        di_ass_final = torch.mean(torch.stack(l_di_ass, dim = 1), dim = 1)

        dr_sample = dr_fusion_final[sample[:, 0]]
        di_sample = di_fusion_final[sample[:, 1]]

        m_result = multiple_operator(dr_sample, di_sample)
        r_result = rotate_operator(dr_sample, di_sample)
        drdi_embedding = torch.cat([dr_sample, di_sample, m_result, r_result], dim = 1)
        
        output = self.mlp(drdi_embedding)

        return output, (dr_fusion_final, di_fusion_final), (dr_sim_final, di_sim_final), (dr_ass_final, di_ass_final)
    
    def load(self, checkpoint_dir):
        self.load_state_dict(torch.load(checkpoint_dir))