import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
epoch = 150
Lr = 0.02
def to_near_half(a):
    return torch.round(a * 2) / 2
def accuracy(a, b, threshold = 0.5):
    return sum(abs(a - b) <= threshold)/ len(a)
class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.alpha = torch.nn.Parameter(torch.zeros(1000))
        self.beta = torch.nn.Parameter(torch.zeros(10000))
        self.gamma = torch.nn.Parameter(torch.zeros(24))
        self.g = torch.nn.Parameter(torch.tensor(1.0))
        self.f1 = torch.nn.Linear(73,64)
        self.f2 = torch.nn.Linear(64,32)
        self.res = torch.nn.Linear(32,32)
        self.f6 = torch.nn.Linear(32,1)
        self.dropout = torch.nn.Dropout(p=0.12,inplace = False)
        self.relu = torch.nn.ReLU()
        self.gelu = torch.nn.GELU()
        
        
    def forward(self, x):
        # return torch.ones_like(x, dtype = float) * 3.5
        x = x.clone()
        a = x[:,0].int()
        b = x[:,1].int()
        x[:,0] = self.alpha[a]
        x[:,1] = self.beta[b]
        for i in range(3, 23):
            x[:, i] *= self.gamma[i]
        x = self.gelu(self.f1(x))
        x = self.gelu(self.f2(x))
        x = self.dropout(x)#dropout策略
        x = self.gelu(self.res(x) + x)#残差网络
        return self.f6(x) + self.g

def one_test(which, x_train, y_train, x_test, y_test):
    Q = MyNet()
    optimizer = torch.optim.Adam(Q.parameters(), lr = Lr, weight_decay=0.0055)
    scheduler = StepLR(optimizer=optimizer, step_size= 12, gamma = 0.9, last_epoch = -1)
    loss_f = F.mse_loss
    loss_table = []
    loss_table1 = []
    for _ in range(epoch):
        y_predict = Q(x_train)[:,0]
        loss = loss_f(y_predict, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if _ > 80:
            loss_table.append(float(torch.mean(loss_f(Q(x_train)[:,0], y_train))))
            loss_table1.append(float(torch.mean(loss_f(Q(x_test)[:,0], y_test))))
    plt.cla()
    plt.plot(loss_table)
    plt.plot(loss_table1)
    plt.savefig("fig.png")
    test_loss = loss_f(Q(x_train)[:,0], y_train)
    test_loss1 = loss_f(Q(x_test)[:,0], y_test)
    acu = accuracy(to_near_half(Q(x_test)[:, 0]), y_test)
    print("第%d折训练集均方损失为:%f,测试集均方损失为:%f,准确率为:%f"%(which, test_loss, test_loss1, acu))

if __name__ == "__main__":
    kf=KFold(n_splits=10, shuffle = True)
    which=0#折数
    df = torch.tensor(np.array(pd.read_csv("cb_table2.csv"))[:,1:]).float()
    x = kf.split(df)
    for train,test in kf.split(df):#对于十折中每一套数据都进行一次测试，最终取平均值
        x_train = df[train,:-1]
        y_train = df[train,-1]
        x_test = df[test,:-1]
        y_test = df[test,-1]
        which+=1
        one_test(which, x_train, y_train, x_test, y_test)