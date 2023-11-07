"""我们将从零开始实现整个方法， 包括数据流水线、模型、损失函数和小批量随机梯度下降优化器"""
import torch
import random
from d2l import torch as d2l


"""生成数据集"""

def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1,1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
print(features[0], '\n', labels[0])


d2l.set_figsize()
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
d2l.plt.show()


"""读取数据集"""
batch_size = 10
def data_iter(features, lables, batch_size):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        indices_batch = torch.tensor(indices[i : min(num_examples, i + batch_size)])
        yield features[indices_batch], lables[indices_batch]

for X, y in data_iter(features, labels, batch_size):
    print(X, '\n', y)
    break

"""定义模型初始化参数"""
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

"""定义模型"""
def linreg(X, w, b):
    return torch.matmul(X, w) + b
"""定义损失函数"""
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape))**2 / 2
"""定义优化方法"""
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()



"""训练模型"""
lr = 0.03
net = linreg
loss = squared_loss
num_epoch = 3

for epoch in range(num_epoch):
    for X, y in data_iter(features, labels, batch_size):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch{epoch + 1}, loss{float(train_l.mean()):f}')


print(f'w的估计误差：{true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差：{true_b - b}')


