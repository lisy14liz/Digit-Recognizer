import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.autograd import Variable
from torchvision.transforms import *
import math
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset


# 导入数据
class M_dataset(Dataset):
    def __init__(self, file_path, istrain, p=1,transform=transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                                                         transforms.Normalize(mean=(0.5,), std=(0.5,))])
                 ):
        df = pd.read_csv(file_path)

        if istrain:
            if(p==1):
                self.X = df.iloc[:, 1:].values.reshape((-1, 28, 28)).astype(np.uint8)[:, :, :, None]
                self.y = torch.from_numpy(df.iloc[:, 0].values)
            else:
                #划分验证集
                df=df.sample(frac=p)
                self.X = df.iloc[:, 1:].values.reshape((-1, 28, 28)).astype(np.uint8)[:, :, :, None]
                self.y = torch.from_numpy(df.iloc[:, 0].values)

        else:
            self.X = df.values.reshape((-1, 28, 28)).astype(np.uint8)[:, :, :, None]
            self.y = None
        self.transform = transform

        print(df.shape)
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.transform(self.X[idx]), self.y[idx]
        else:
            return self.transform(self.X[idx])


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=16,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,  # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # 在 2x2 空间里向下采样, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 3, 1, 1),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)  # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output

# 训练模型
def train_model(epoch,data_loader):
    for step, (b_x, b_y) in enumerate(data_loader):  # 分配 batch data, normalize x when iterate train_loader
        b_x, b_y = Variable(b_x), Variable(b_y)
        if torch.cuda.is_available():
            b_x = b_x.cuda()
            b_y = b_y.cuda()
        output = cnn(b_x)  # cnn output
        loss = loss_func(output, b_y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        # if (step + 1) % 100 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, (step + 1) * len(b_x), len(data_loader.dataset),
        #                100. * (step + 1) / len(data_loader), loss.item()))
# 得到在数据集上的准确率
def evaluate(data_loader):
    cnn.eval()
    loss = 0
    correct = 0

    for data, target in data_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        output = cnn(data)
        loss += F.cross_entropy(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    loss /= len(data_loader.dataset)

    # print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
    #     loss, correct, len(data_loader.dataset),
    #     float(correct) * 100.0 / len(data_loader.dataset)))
    return float(correct) * 100.0 / len(data_loader.dataset)

# 得到在数据集上的预测结果
def prediciton(data_loader):

    cnn.eval()
    test_pred = torch.LongTensor()

    for i, data in enumerate(data_loader):
        data = Variable(data, volatile=True)
        if torch.cuda.is_available():
            data = data.cuda()
        output = cnn(data)
        pred = output.cpu().data.max(1, keepdim=True)[1]
        test_pred = torch.cat((test_pred, pred), dim=0)

    return test_pred


if __name__ == '__main__':
    #划分验证集进行训练
    train_d = M_dataset('train.csv', 1, p=0.8)
    validation_d=M_dataset('train.csv', 1, p=0.2)
    test_d = M_dataset('test.csv', 0)
    torch.manual_seed(1)
    # Hyper Parameters
    EPOCH = 100 # 最大迭代次数
    BATCH_SIZE = 50
    LR = 0.003  # 学习率
    # 批训练
    train_loader = torch.utils.data.DataLoader(dataset=train_d, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_d, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_d, batch_size=BATCH_SIZE, shuffle=False)
    cnn = CNN()
    cnn.train()

    # print(cnn)  # net architecture

    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

    if torch.cuda.is_available():
        cnn = cnn.cuda()
        loss_func = loss_func.cuda()
    # training and testing
    max_accuracy=0
    count=0
    all_accuracy=[]
    end_epoch=EPOCH
    for epoch in range(EPOCH):
        train_model(epoch, train_loader)
        accuracy = evaluate(validation_loader)
        all_accuracy.append(accuracy)
        if count == 10:
            print("end epoch: ",epoch + 1,'max_accuracy=',max_accuracy)
            end_epoch=epoch+1
            break
        if accuracy <= max_accuracy:
            count=count+1
        else:
            max_accuracy=accuracy
            #print('epoch=',epoch,'max_accuracy=',max_accuracy)
            count=0
    plt.figure()

    plt.plot(list(range(len(all_accuracy))),all_accuracy)
    plt.show()

    #用全部数据进行训练
    train_d = M_dataset('train.csv', 1)
    test_d = M_dataset('test.csv', 0)
    torch.manual_seed(1)
    # Hyper Parameters
    EPOCH = end_epoch-10
    # 批训练
    train_loader = torch.utils.data.DataLoader(dataset=train_d, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_d, batch_size=BATCH_SIZE, shuffle=False)
    cnn = CNN()
    cnn.train()
    # print(cnn)  # net architecture

    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
    if torch.cuda.is_available():
        cnn = cnn.cuda()
        loss_func = loss_func.cuda()
    # training and testing
    for epoch in range(EPOCH):
        train_model(epoch, train_loader)
        evaluate(train_loader)
    test_pred = prediciton(test_loader)
    out_df = pd.DataFrame(np.c_[np.arange(1, len(test_d) + 1)[:, None], test_pred.numpy()],
                          columns=['ImageId', 'Label'])
    out_df.head()
    out_df.to_csv('submission.csv', index=False)
