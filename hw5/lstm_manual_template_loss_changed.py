# 用Pytorch手写一个LSTM网络，在IMDB数据集上进行训练

import os
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from utils import load_imdb_dataset, Accuracy
import sys

from utils import load_imdb_dataset, Accuracy
from tqdm import tqdm
import math

torch.autograd.set_detect_anomaly(True)

use_mlu = False
# try:
#     import torch_mlu
#     import torch_mlu.core.mlu_model as ct
#     global ct
#     use_mlu = torch.mlu.is_available()
# except:
#     use_mlu = False

if use_mlu:
    device = torch.device('mlu:0')
else:
    print("MLU is not available, use GPU/CPU instead.")
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

X_train, y_train, X_test, y_test = load_imdb_dataset('data', nb_words=20000, test_split=0.2)

seq_Len = 200
vocab_size = len(X_train) + 1


class ImdbDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):

        data = self.X[index]
        data = np.concatenate([data[:seq_Len], [0] * (seq_Len - len(data))]).astype('int32')  # set
        label = self.y[index]
        return data, label

    def __len__(self):

        return len(self.y)


# 你需要实现的手写LSTM内容，包括LSTM类所属的__init__函数和forward函数
class LSTM(nn.Module):
    '''
    手写lstm，可以用全连接层nn.Linear，不能直接用nn.LSTM
    '''

    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM层
        # 初始权重(自己注册的要更新权重，要定义为self的属性，并且用nn.Parameter()括起来)
        k = math.sqrt(1 / self.hidden_size)
        self.weight_ih = nn.Parameter(2 * k * torch.rand(self.num_layers, 4 * self.hidden_size, self.input_size) - k)
        self.weight_hh = nn.Parameter(2 * k * torch.rand(self.num_layers, 4 * self.hidden_size, self.hidden_size) - k)
        self.bias_ih = nn.Parameter(2 * k * torch.rand(self.num_layers, 4 * self.hidden_size) - k)
        self.bias_hh = nn.Parameter(2 * k * torch.rand(self.num_layers, 4 * self.hidden_size) - k) 
        # 沿层方向更新

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.transpose(0, 1) # (seq_len, batch_size, input_size)

        # 初始隐藏状态和细胞状态
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device) # 既沿层方向更新，也沿序列方向更新
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device) # 沿序列方向更新

        # 计算
        h_t_minus_1,c_t_minus_1 = h_0, c_0
        output = []
        for token in range(seq_len):
            x_t = x[token] # 每层的输入为上一层的隐藏状态，记得更新
            h_next, c_next = [], []
            for layer in range(self.num_layers):
                gates = (h_t_minus_1[layer] @ self.weight_hh[layer].transpose(0, 1)
                        + self.bias_hh[layer]
                        + x_t @ self.weight_ih[layer].transpose(0, 1) 
                        + self.bias_ih[layer])
                f_t, i_t, c_bar_t, o_t = torch.chunk(gates, 4, dim=1)
                # (batch_size, hidden_size)

                new_c = c_t_minus_1[layer] * torch.sigmoid(f_t) \
                            + torch.sigmoid(i_t) * torch.tanh(c_bar_t)
                
                new_h = torch.tanh(new_c) * torch.sigmoid(o_t)
                x_t = new_h
                c_next.append(new_c)
                h_next.append(new_h)

            output.append(h_next[-1])
            h_t = torch.stack(h_next)
            c_t = torch.stack(c_next)
            h_t_minus_1 = h_t
            c_t_minus_1 = c_t
        c_n = c_t
        h_n = h_t
        # output维度是(seq_len, batch_size, hidden_size),取[-1]即最后一个token的输出
        return output[-1], (h_n, c_n)


# 你需要实现网络推理和训练内容，仅需要完善forward函数
class Net(nn.Module):
    '''
    一层LSTM的文本分类模型
    '''

    def __init__(self, embedding_size=64, hidden_size=64, num_layers=1, num_classes=2):
        super(Net, self).__init__()

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # LSTM层
        self.lstm = LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers)
        # 全连接层
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        '''
        x: 输入, shape: (batch_size, seq_len)
        '''

        # 词嵌入层
        x = self.embedding(x)
        # LSTM层
        x, _ = self.lstm(x)
        # 全连接层
        x = self.fc1(x)
        x = self.fc2(x)
        # 分类层
        # x = torch.nn.functional.softmax(x, dim=1) # softmax函数不是对单个元素的，调用比sigmoid，log长一点
        return x

n_epoch = 10
batch_size = 64
print_freq = 2

train_dataset = ImdbDataset(X=X_train, y=y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = ImdbDataset(X=X_test, y=y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

net = Net()
metric = Accuracy()
print(net)


def train(model, device, train_loader, optimizer, epoch):
    model = model.to(device)
    model.train()
    loss_func = torch.nn.BCEWithLogitsLoss(reduction="mean")
    train_correct = 0
    train_loss = 0
    train_total = 0
    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, target = data.to(device), target.float().to(device)
        optimizer.zero_grad()
        output = model(data).squeeze()
        # loss = F.nll_loss(output, target)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()

        preds = (torch.sigmoid(output) > 0.5).long()
        train_correct += (preds == target.long()).sum().item()
        train_total += target.size(0)
        train_loss += loss.item()

    acc = train_correct / train_total
    avg_loss = train_loss / len(train_loader)
    print('Train Epoch: {} Loss: {:.6f} \t Acc: {:.6f}'.format(epoch, avg_loss, acc))


def test(model, device, test_loader):
    model = model.to(device)
    model.eval()
    loss_func = torch.nn.BCEWithLogitsLoss(reduction="mean")
    test_loss = 0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.float().to(device)
            output = model(data).squeeze()
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += loss_func(output, target).item()

            preds = (torch.sigmoid(output) > 0.5).long()
            test_correct += (preds == target.long()).sum().item()
            test_total += target.size(0)

    acc = test_correct / test_total
    avg_loss = test_loss / len(test_loader)
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(avg_loss, acc))


optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0)
gamma = 0.7
for epoch in range(1, n_epoch + 1):
    train(net, device, train_loader, optimizer, epoch)
    test(net, device, test_loader)
