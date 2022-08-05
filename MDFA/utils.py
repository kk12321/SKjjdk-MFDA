import numpy as np
import pandas as pd
import scipy.sparse as sp
import h5py
import torch
from torch.utils.data import Dataset
from numpy.linalg import inv
from scipy.linalg import fractional_matrix_power
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

def draw_loss(all_train_losses, all_valid_losses,name,epochs):
    plt.figure()
    # 绘制训练损失曲线
    plt.plot(all_train_losses, label="Train Loss")
    # 绘制验证损失曲线, 颜色为红色
    plt.plot(all_valid_losses, color="red", label="Valid Loss")
    # 定义横坐标刻度间隔对象, 间隔为1, 代表每一轮次
    # x_major_locator = MultipleLocator(1)
    # # 获得当前坐标图句柄
    # ax = plt.gca()
    # # 设置横坐标刻度间隔
    # ax.xaxis.set_major_locator(x_major_locator)
    # 设置横坐标取值范围
    plt.xlim(1, epochs)
    # 曲线说明在左上方
    plt.legend(loc='upper left')
    # 保存图片
    plt.savefig("./loss_only_view_{}_{}.png".format(name,epochs))

def load_true_adj(adj_file):
    ddi_label = pd.read_csv(adj_file, dtype=int,header=None)
    ddi_arr = ddi_label.iloc[:, [0, 1]].values
    drug_a = ddi_label.iloc[:, 0].values
    drug_b = ddi_label.iloc[:, 1].values

    label = ddi_label.iloc[:, 2].values

    return ddi_arr,label,drug_a,drug_b

def load_knn_adj(adj_knn):
    ddi_knn = pd.read_csv(adj_knn, dtype=int,header=None).values
    return ddi_knn

def load_graph(adj_file):
    # load ddi_matrix to get ddi_array and label

    ddi_arr,_,_,_=load_true_adj(adj_file)

    adj = sp.coo_matrix((np.ones( ddi_arr.shape[0]), ( ddi_arr[:, 0], ddi_arr[:, 1])),
                        shape=(572, 572), dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj

###load diff_graph
def load_graph_diff(adj_file):
    ddi_arr,_,_,_=load_true_adj(adj_file)
    adj = sp.coo_matrix((np.ones( ddi_arr.shape[0]), ( ddi_arr[:, 0], ddi_arr[:, 1])),
                        shape=(572, 572), dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    # adj = normalize(adj)
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    # adj=adj.to_dense()
    adj=adj.toarray()
    return adj

def diff(adj, alpha):
    d = np.diag(np.sum(adj, 1))
    dinv = fractional_matrix_power(d, -0.5)
    at = np.matmul(np.matmul(dinv, adj), dinv)
    adj_1 = alpha * inv((np.eye(adj.shape[0]) - (1 - alpha) * at))
    adj_last = sp.coo_matrix(adj_1) #coo_matrix
###将coo_matrix变成sparse_tensor
    values = adj_last.data
    indices = np.vstack((adj_last.row, adj_last.col))
    i = torch.LongTensor(indices)
    v = torch.LongTensor(values)
    edge_idx = torch.sparse_coo_tensor(i, v, adj_last.shape)
    return edge_idx




def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class load_data(Dataset):
    def __init__(self, dataset):
        # #use self_attention getting STE_feature.txt
        # DF=pd.read_csv('data/ATT_and_DMDDI_result/STE_self_and_attention_1716.csv', dtype=float, header=None, index_col=0)
        # DF=DF.values
        # print("STE.shape",DF.shape)
        # self.x=DF

        self.x = np.loadtxt('./data/STE_feature.txt', dtype=float)
###when test different ddi class dateset,need to introduce  different file "ddi_class_X.csv"
        # y = pd.read_csv('./graph/ddi_class_3.csv', dtype=int)
        # label=y.iloc[:,2].values # (33214,)
        # self.y=label


    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])),\
               torch.from_numpy(np.array(self.y[idx])),\
               torch.from_numpy(np.array(idx))






