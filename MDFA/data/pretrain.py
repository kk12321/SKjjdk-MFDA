
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.nn import Linear
from torch.utils.data import Dataset
from sklearn.cluster import KMeans


#torch.cuda.set_device(3)

# model = AE(
#         n_enc_1=500,
#         n_enc_2=2000,
#         n_enc_3=256,
#         n_input=1716,
#         n_z=65
#          )

class AE(nn.Module):
    #256,2000
    def __init__(self, n_enc_1, n_enc_2, n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)  #1716->2000
        self.enc_2 = Linear(n_enc_1, n_enc_2)  #2000->256
        self.z_layer = Linear(n_enc_2, n_z)    #256->65

        self.dec_1 = Linear(n_z,n_enc_2)  #65->256
        self.dec_2 = Linear(n_enc_2, n_enc_1) #256->2000
        self.x_bar_layer = Linear(n_enc_1, n_input)  # 2000->1716

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x)) #2000
        enc_h2 = F.relu(self.enc_2(enc_h1))
        # enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h2)
        dec_h1 = F.relu(self.dec_1(z))  # 解码
        dec_h2 = F.relu(self.dec_2(dec_h1))
        # dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h2)
        return x_bar, z


class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001 * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def pretrain_ae(model, dataset):
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    print("pretrain_ae中model:",model)
    optimizer = Adam(model.parameters(), lr=1e-3)
    for epoch in range(30):
        # adjust_learning_rate(optimizer, epoch)
        for batch_idx, (x, _) in enumerate(train_loader):
            # x = x.cuda()
            x_bar, _ = model(x)
            loss = F.mse_loss(x_bar, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            x = torch.Tensor(dataset.x).float()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            print('{} loss: {}'.format(epoch, loss))

        if epoch == 29:
            # X_Z= np.array(z.detach().numpy())
            # np.savetxt('z_class572.csv',X_Z)
             torch.save(model.state_dict(), 'multi_head_1716_520_260_65.pkl')


model = AE(
        n_enc_1=520,
        n_enc_2=260,
        n_input=1716,
        n_z=65
         )

x = np.loadtxt('STE_feature.txt', dtype=float)
print(x.shape)
# y = np.loadtxt('dblp_label.txt', dtype=int)

dataset = LoadDataset(x)
pretrain_ae(model, dataset)
