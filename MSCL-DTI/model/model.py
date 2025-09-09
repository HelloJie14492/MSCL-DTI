import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
from .Informer import AttentionLayer, ProbAttention


class EncoderLayer(nn.Module):
    def __init__(self, i_channel, o_channel, growth_rate, groups, pad2=7):
        super(EncoderLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=i_channel, out_channels=o_channel, kernel_size=(2 * pad2 + 1), stride=1, groups=groups, padding=pad2, bias=False)
        self.bn1 = nn.BatchNorm1d(i_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(in_channels=o_channel, out_channels=growth_rate, kernel_size=(2 * pad2 + 1), stride=1, groups=groups, padding=pad2, bias=False)
        self.bn2 = nn.BatchNorm1d(o_channel)
        self.drop_rate = 0.1

    def forward(self, x):
        # xn = self.bn1(x)
        xn = self.relu(x)
        xn = self.conv1(xn)
        xn = self.bn2(xn)
        xn = self.relu(xn)
        xn = self.conv2(xn)
        return torch.cat([x, xn], 1)


class Encoder(nn.Module):
    def __init__(self, inc, outc, growth_rate, layers, groups, pad1=15, pad2=7):
        super(Encoder, self).__init__()
        self.layers = layers
        self.relu = nn.ReLU(inplace=True)
        self.conv_in = nn.Conv1d(in_channels=inc, out_channels=inc, kernel_size=(pad1 * 2 + 1), stride=1, padding=pad1, bias=False)
        self.dense_cnn = nn.ModuleList(
            [EncoderLayer(inc + growth_rate * i_la, inc + (growth_rate // 2) * i_la, growth_rate, groups, pad2) for i_la in range(layers)])
        self.conv_out = nn.Conv1d(in_channels=inc + growth_rate * layers, out_channels=outc, kernel_size=(pad1 * 2 + 1), stride=1, padding=pad1, bias=False)

    def forward(self, x):
        x = self.conv_in(x)
        for i in range(self.layers):
            x = self.dense_cnn[i](x)
        x = self.relu(x)
        x = self.conv_out(x)
        x = self.relu(x)
        return x


class DT_LeNet(nn.Module):
    def __init__(self, hidden, dropout, classes, layers):
        super(DT_LeNet, self).__init__()
        self.CNNs = nn.ModuleList([nn.Conv1d(in_channels=hidden, out_channels=hidden, kernel_size=7, padding=3) for _ in range(layers)])
        self.BN = nn.BatchNorm1d(hidden)  # nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(layers)])
        self.FC_combs = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.FC_down = nn.Linear(hidden, 128)
        self.FC_out = nn.Linear(128, classes)
        self.layers = layers
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.05)

    def forward(self, dti_feature):
        dti_feature = dti_feature.permute(0, 2, 1)  # self.BN(dti_feature.permute(0, 2, 1))
        for i in range(self.layers):
            dti_feature = self.act(self.CNNs[i](dti_feature)) + dti_feature
        dti_feature = dti_feature.permute(0, 2, 1)
        dti_feature = torch.mean(dti_feature, dim=1)
        GRL_feature = dti_feature.clone()
        for i in range(self.layers):
            dti_feature = self.act(self.FC_combs[i](dti_feature))
        dti_feature = self.FC_down(dti_feature)
        dti = self.FC_out(dti_feature)
        return dti, dti_feature, GRL_feature


class GRL(nn.Module):
    def __init__(self, max_iter):
        super(GRL, self).__init__()
        self.iter_num = 0
        self.alpha = 10
        self.max_iter = max_iter

    def forward(self, input):
        self.iter_num += 1
        return input * 1.0

    def backward(self, gradOutput):
        coeff = np.float(2.0 / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter)) - 1)
        return -coeff * gradOutput

class Discriminator(nn.Module):
    def __init__(self, max_iter, source_number, device):
        super(Discriminator, self).__init__()
        self.fc1 = Parameter(torch.Tensor(256, 128))
        self.fc2 = Parameter(torch.Tensor(75, 128))
        self.fc3 = Parameter(torch.Tensor(128, source_number))
        self.drop_lm = nn.Dropout(0.0)
        self.drop_sty = nn.Dropout(0.0)
        self.drop = nn.Dropout(0.5)
        self.grl_layer = GRL(10000)
        self.grl_layer2 = GRL(100)
        self.source_number = source_number
        self.device = device

    def forward(self, feature1, feature2):
        if self.source_number > 2:
            adversarial_out1 = self.grl_layer(self.drop_lm(feature1.detach()))
            adversarial_out2 = self.grl_layer2(self.drop_sty(feature2))
            adversarial_out1 = torch.matmul(adversarial_out1,  nn.init.xavier_uniform_(self.fc1))
            adversarial_out2 = torch.matmul(adversarial_out2, nn.init.xavier_uniform_(self.fc2))
            adversarial_out = adversarial_out2 * torch.sigmoid(adversarial_out1)
            adversarial_out = torch.matmul(self.drop(torch.relu(adversarial_out)), nn.init.xavier_uniform_(self.fc3))
            # print(adversarial_out)
        else:
            adversarial_out = torch.zeros((feature1.shape[0], 1), device=self.device)
        return adversarial_out


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.3, source_number=2):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.source_number = source_number

    def forward(self, features, labels):
        """计算监督对比损失
        Args:
            features: (batch, dim) 归一化后的特征向量
            labels: (batch,) 样本标签
        """
        if self.source_number <= 2:
            return torch.tensor(0.0, device=features.device)

        device = features.device
        batch_size = features.size(0)
        labels = labels.contiguous().view(-1, 1)

        # 创建正样本掩码（对角线置0）
        mask = torch.eq(labels, labels.T).float().to(device)
        mask.fill_diagonal_(0)

        # 检查有效样本
        valid = mask.sum(1) > 0
        if not valid.any():
            return torch.tensor(0.0, device=device)

        # 计算相似度
        anchor_dot_contrast = torch.matmul(features, features.T) / self.temperature
        logits_max, _ = anchor_dot_contrast.max(dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 排除自身对比
        logits_mask = 1 - torch.eye(batch_size, device=device)
        exp_logits = torch.exp(logits) * logits_mask

        # 计算对数概率
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # 计算正样本的平均对数概率
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

        # 仅对有效样本求平均
        loss = -mean_log_prob_pos[valid].mean()
        return loss


class Smooth_loss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(Smooth_loss, self).__init__()
        self.smoothing = smoothing

    def forward(self, logits, labels):
        confidence = 1 - self.smoothing
        logprobs = F.log_softmax(logits, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class Fusion(nn.Module):
    def __init__(self, hidden1, hidden2, dropout=0.05):
        super(Fusion, self).__init__()
        self.si_L = nn.Sigmoid()
        self.si_S = nn.Sigmoid()
        self.so_f = nn.Sigmoid()
        self.combine = nn.Linear(128 * 4, 128)
        self.ln = nn.LayerNorm(128)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, LM_fea, Sty_fea):

        Sty_fea_norm = Sty_fea * (abs(torch.mean(LM_fea))/abs(torch.mean(Sty_fea)))
        f_h = torch.cat((LM_fea.unsqueeze(1), Sty_fea_norm.unsqueeze(1)), dim=1)
        f_att = torch.mean(f_h, dim=1)
        f_att = self.so_f(f_att)
        fus_fea = torch.cat((LM_fea, Sty_fea, LM_fea * f_att, Sty_fea * f_att), dim=1)
        fus_fea = self.combine(fus_fea)
        return fus_fea

class DGMM_DTI(nn.Module):
    def __init__(self, layer_gnn, device, source_number, hidden1=256, hidden2=75, n_layers=1, attn_heads=1,
                 dropout=0.0):
        super(DGMM_DTI, self).__init__()
        '''GNN'''
        self.embed_protein = nn.Embedding(26, hidden2)
        self.W_dnn = nn.ModuleList([nn.Linear(hidden2, hidden2)
                                    for _ in range(layer_gnn)])
        self.W_pnn = nn.ModuleList([nn.Linear(hidden2, hidden2)
                                    for _ in range(layer_gnn)])

        self.gnn_act = nn.GELU()
        self.G_A = nn.ModuleList(
            [nn.Conv1d(in_channels=hidden2, out_channels=hidden2, kernel_size=3, padding=1, groups=hidden2, bias=False)
             for _ in range(layer_gnn)])
        self.encoder_protein_GNN = Encoder(hidden2, hidden2, 15, 5, groups=1, pad1=15, pad2=7)
        self.bn_A = nn.ModuleList([nn.BatchNorm1d(hidden2) for _ in range(layer_gnn)])
        self.bn_B = nn.ModuleList([nn.BatchNorm1d(hidden2) for _ in range(layer_gnn)])
        self.bn_C = nn.ModuleList([nn.BatchNorm1d(hidden2) for _ in range(layer_gnn)])
        self.bn_D = nn.ModuleList([nn.BatchNorm1d(hidden2) for _ in range(layer_gnn)])
        self.gnn_drop = nn.Dropout(p=0.05)
        self.gnn_output = nn.Linear(hidden2, hidden2, bias=False)

        '''LM'''
        self.encoder_protein_LM = Encoder(1024, hidden1, 128, 3, groups=64, pad1=7, pad2=3)
        self.encoder_drug = Encoder(768, hidden1, 128, 3, groups=32, pad1=7, pad2=3)
        self.Informer_blocks = nn.ModuleList(
            [AttentionLayer(ProbAttention(None, 3, 0), hidden1, attn_heads),
             AttentionLayer(ProbAttention(None, 5, 0), hidden1, attn_heads)])

        '''DECISION'''
        self.soft_1 = nn.Softmax(-1)
        self.soft_2 = nn.Softmax(-1)
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.layer_gnn = layer_gnn
        self.hidden = hidden1
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.fusion = Fusion(hidden1, hidden2)
        self.FC_out1 = DT_LeNet(hidden1, 0.05, 2, 3)
        self.FC_out2 = DT_LeNet(hidden2, 0.0, 2, 3)
        self.DTI_feature = nn.ModuleList([nn.Linear(128, 128) for _ in range(2)])
        self.act = nn.ReLU()
        self.DTI_Pre = nn.Linear(128, 2)
        self.dis = Discriminator(100, source_number, device)
        self.source_number = source_number
        self.lamda = Parameter(torch.Tensor([0.8, 0.1, 0.1]))
        self.SupConloss = SupervisedContrastiveLoss(temperature=0.3, source_number=source_number)


    def Style_Exract(self, df, da, pf, layer):
        for i in range(layer):
            ds = self.gnn_act(self.W_dnn[i](df))
            ps = self.gnn_act(self.W_pnn[i](pf))
            dg_A = self.bn_A[i](self.G_A[i](ps.permute(0, 2, 1))).permute(0, 2, 1)
            G_CB = torch.matmul(pf, df.permute(0, 2, 1))
            dg_B = self.bn_B[i](torch.matmul(self.soft_1(G_CB), ds).permute(0, 2, 1)).permute(0, 2, 1)
            dg_C = self.bn_C[i](torch.matmul(da, ds).permute(0, 2, 1)).permute(0, 2, 1)
            G_BC = torch.matmul(df, pf.permute(0, 2, 1))
            dg_D = self.bn_D[i](torch.matmul(self.soft_2(G_BC), ps).permute(0, 2, 1)).permute(0, 2, 1)
            pf = dg_A + dg_B + pf
            df = dg_C + dg_D + df
        dt = torch.cat((pf, df), dim=1)
        return dt

    def forward(self, inputs):

        molecule_smiles, molecule_atoms, molecule_adjs, proteins, protein_LM, molecule_LM = inputs
        N = molecule_smiles.shape[0]

        """DTI 1D feature with pretrain language model"""
        proteins_acids_LM = self.encoder_protein_LM(protein_LM.permute(0, 2, 1)).permute(0, 2, 1)  # .mean(dim=1)
        molecule_smiles_LM = self.encoder_drug(molecule_LM.permute(0, 2, 1)).permute(0, 2, 1)  # .mean(dim=1)

        DT_1D_Feature = torch.cat((proteins_acids_LM, molecule_smiles_LM), 1)
        DT_1D_P_att = self.dropout(self.Informer_blocks[0](DT_1D_Feature, proteins_acids_LM, proteins_acids_LM, None))
        DT_1D_D_att = self.dropout(self.Informer_blocks[1](DT_1D_Feature, molecule_smiles_LM, molecule_smiles_LM, None))
        DT_1D_Feature = DT_1D_P_att + DT_1D_D_att

        """DTI 2D feature with Graph Nerual Networks"""
        proteins_acids_GNN = torch.zeros((proteins.shape[0], proteins.shape[1], 75), device=self.device)
        DT_2D_Feature = torch.zeros((N, 1300, 75), device=self.device) # b 1300 d 1400
        for i in range(N):
            proteins_acids_GNN[i, :, :] = self.embed_protein(torch.LongTensor(proteins[i].to('cpu').numpy()).cuda())
        proteins_acids_GNN = self.encoder_protein_GNN(proteins_acids_GNN.permute(0, 2, 1)).permute(0, 2, 1)
        DT_2D_F = self.Style_Exract(molecule_atoms, molecule_adjs, proteins_acids_GNN, self.layer_gnn)
        t = DT_2D_F.shape[1]
        if t < 1300:
            DT_2D_Feature[:, 0:t, :] = DT_2D_F
        else:
            DT_2D_Feature = DT_2D_F[:, 0:1300, :]

        # DT_2D_Feature = DT_2D_P_att + DT_2D_D_att

        """Combine the features of two modals"""
        dti1d, dti1d_feature, LM_feature = self.FC_out1(DT_1D_Feature)
        dti2d, dti2d_feature, GRL_feature = self.FC_out2(DT_2D_Feature)
        dis_invariant = self.dis(LM_feature, GRL_feature)
        DTI = self.fusion(dti1d_feature, dti2d_feature)
        DTI_normal = DTI.clone()
        for i in range(2):
            DTI = self.act(self.DTI_feature[i](DTI))
        DTI = self.DTI_Pre(DTI)
        lam_DTI = self.lamda[0] * DTI.detach() + self.lamda[1] * dti1d.detach() + self.lamda[2] * dti2d.detach()
        return DTI, dti1d, dti2d, dti1d_feature, dti2d_feature, DTI_normal, lam_DTI, self.lamda, dis_invariant

    def __call__(self, data, epoch=1, train=True):

        inputs, correct_interaction, SID = data[:-2], data[-2], data[-1]
        correct_interaction = torch.LongTensor(correct_interaction.to('cpu').numpy()).cuda()
        SID = torch.LongTensor(SID.to('cpu').numpy()).cuda()
        LACE = Smooth_loss()
        protein_drug_interaction, dti1d, dti2d, dti1d_feature, dti2d_feature, DTI_normal, lam_DTI, lamda, dis_invariant = self.forward(inputs)  # , dis_invariant
        if train:
            contrast_label = correct_interaction
            loss1 = F.cross_entropy(protein_drug_interaction, correct_interaction)
            loss2 = F.cross_entropy(dti1d, correct_interaction)
            loss3 = F.cross_entropy(dti2d, correct_interaction)
            loss4 = LACE(dis_invariant, SID)
            loss5 = self.SupConloss(F.normalize(DTI_normal, p=2, dim=1), contrast_label)

            return loss1 * 1 + loss2 * 1 + loss3 * 1.25 + loss4 * 1 + loss5 * 1
        else:
            correct_labels = correct_interaction
            ys1 = F.softmax(protein_drug_interaction * 0.4 + dti1d * 0.27 + dti2d * 0.33, 1)
            # ys1 = F.softmax(protein_drug_interaction * 0.4 + dti1d * 0.35 + dti2d * 0.25, 1)
            ys2 = F.softmax(dti1d, 1)
            ys3 = F.softmax(dti2d, 1)
            return correct_labels, ys1, ys2, ys3, dti1d_feature, dti2d_feature, DTI_normal

