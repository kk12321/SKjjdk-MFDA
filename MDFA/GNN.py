import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
class GNNLayer(Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True):
        features =  features.to(torch.float32)
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        if active:
            output = F.relu(output)
        return output


class GATLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features):
        super(GATLayer, self).__init__()
        self.dropout = 0.6
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = 0.2

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj,concat=True): #()
        Wh = torch.mm(h, self.W)  # (572,65)
        e = self._prepare_attentional_mechanism_input(Wh)  # 设的注意力系数

        zero_vec = -9e15 * torch.ones_like(e)
        adj=adj.to_dense() #把tensor张量变成数组形式
        # print(adj[:5])
        attention = torch.where(adj > 0, e, zero_vec)#当原位置有则不变，否则变为负无穷大
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)  # （2708，2708）
        ###可以将注意力机制进行输出
        h_prime = torch.matmul(attention, Wh)  # (572,65)
        ###将权重融合之后的进行输出
        if concat:
            return F.elu(h_prime)  # 如果是中间层，有多个注意力，就先得经过激活函数
        else:
            return h_prime,attention

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :]) #(572,1)
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.t()
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

# 原始的参数
# class GraphAttentionLayer(nn.Module):
#     """
#     Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
#     """
#     def __init__(self, in_features, out_features, dropout, alpha, concat=True):
#         super(GraphAttentionLayer, self).__init__()
#         self.dropout = dropout
#         self.in_features = in_features
#         self.out_features = out_features
#         self.alpha = alpha
#         self.concat = concat
#
#         self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
#         nn.init.xavier_uniform_(self.W.data, gain=1.414)
#         self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
#         nn.init.xavier_uniform_(self.a.data, gain=1.414)
#
#         self.leakyrelu = nn.LeakyReLU(self.alpha)
#
#     def forward(self, h, adj):
#         Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features) (2708,8)
#         e = self._prepare_attentional_mechanism_input(Wh) #邓设的注意力系数
#
#         zero_vec = -9e15*torch.ones_like(e)
#         attention = torch.where(adj > 0, e, zero_vec)
#         attention = F.softmax(attention, dim=1)
#         attention = F.dropout(attention, self.dropout, training=self.training)#（2708，2708）
# ###可以将注意力机制进行输出
#         h_prime = torch.matmul(attention, Wh)#真正注意力系数乘  WH
# ###将权重融合之后的进行输出
#         if self.concat:
#             return F.elu(h_prime)#如果是中间层，有多个注意力，就先得经过激活函数
#         else:
#             return h_prime
#
#
#
#
#
#
#     def _prepare_attentional_mechanism_input(self, Wh):
#         # Wh.shape (N, out_feature)
#         # self.a.shape (2 * out_feature, 1)
#         # Wh1&2.shape (N, 1)
#         # e.shape (N, N)
#         Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
#         Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
#         # broadcast add
#         e = Wh1 + Wh2.t()
#         return self.leakyrelu(e)
#
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
