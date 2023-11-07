"""导入所需的包"""

import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

"""生成随机数据集"""
true_w = torch.tensor([2, -3.4])
true_b = torch.tensor(4.2)
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

"""按批量大小读取数据集"""
batch_size = 10
def load_array(data_arrays, batch_size, istrain = True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle = istrain)
data_iter = load_array((features, labels), batch_size)
next(iter(data_iter))

"""模型设计"""

from torch import nn
net = nn.Sequential(nn.Linear(2, 1))

"""初始化模型参数"""

net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

"""定义损失函数"""

loss = nn.MSELoss()

"""定义优化器"""

trainer = torch.optim.SGD(net.parameters(), lr=0.03)

"""训练"""

num_epoch = 3
for epoch in range(num_epoch):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch{epoch + 1}, loss{l:f}')

w = net[0].weight.data
b = net[0].bias.data

print('w的误差：',true_w - w.reshape(true_w.shape))
print('b的误差：',true_b - b.reshape(true_b.shape))