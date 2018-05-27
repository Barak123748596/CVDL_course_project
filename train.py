from config import opt
import torch
from LoadData import train_loader, val_loader
from torch.autograd import Variable
from torch import nn
from torchvision import models
import os
from U_Net import U_Net
import numpy as np
from utils.visualize import Visualizer
from torchnet import meter
from matplotlib import pyplot as plt

EPOCH_NUM = 5
MODEL_PATH = "models/V1.1/U_Net.pkl"
N_CHANNEL = 3
N_CLASS = 2
LR = 2e-5
BATCH_NUM = 36038

# 0: Tesla K20C   1: Quadro 600
print(torch.cuda.is_available())
print(torch.cuda.device_count())


def val(model, dataloader):
    '''
        计算模型在验证集上的准确率等信息
        :param model:
        :param dataloader:
        :return:
        '''
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    confusion_matrix = meter.ConfusionMeter(2)
    for ii, data in enumerate(dataloader):
        input, label = data
        val_input = Variable(input, volatile=True)
        val_label = Variable(label.long(), volatile=True)
        
        if torch.cuda.is_available():
            val_input = val_input.cuda()
            val_label = val_label.cuda()
    
        score = model(val_input)
        # print(np.shape(val_label.cpu().detach().numpy()))
        # print(np.shape(score.cpu().detach().numpy()))

        label_flat = (val_label.view(-1, 1)).squeeze(1)
        score_flat = score.view(-1, 2)
        
        # print(np.shape(label_flat.cpu().detach().numpy()))
        # print(np.shape(score_flat.cpu().detach().numpy()))
        
        confusion_matrix.add(score_flat.cpu().detach(), label_flat.cpu().detach())

    # 把模型恢复为训练模式
    model.train()

    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / \
        (cm_value.sum())
    return confusion_matrix, accuracy


# vis = Visualizer(opt.env)
if os.path.exists(r'models/V1.1/U_Net.pkl'):
    my_model = torch.load('models/V1.1/U_Net.pkl').cuda()
    print("model from load.")
else:
    my_model = U_Net(N_CHANNEL, N_CLASS)
    print("A new model.")
# print(my_model)

criterion = torch.nn.NLLLoss(weight=torch.FloatTensor([0.2, 0.8]))
optimizer = torch.optim.Adam(my_model.parameters(), lr=LR)

if torch.cuda.is_available():
    my_model = my_model.cuda()
    criterion.cuda()

# 统计指标：平滑处理之后的损失，还有混淆矩阵
# loss_meter = meter.AverageValueMeter()
# confusion_matrix = meter.ConfusionMeter(2)
previous_loss = 1e100

train_Loss_record = []
val_Loss_record = []

plt.ion()
plt.title("Loss")

# train
for epoch in range(EPOCH_NUM):
    
    # loss_meter.reset()
    # confusion_matrix.reset()
    
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images)
        labels = Variable(labels.long())
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        labels = labels.squeeze(1)
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = my_model(images)
        # print(np.shape(outputs))
        # print(np.shape(labels))
        loss = criterion(outputs, labels)
        
        # print(loss)
        train_Loss_record.append(loss.cpu().detach().numpy())
        loss.backward()
        optimizer.step()
        
        if i % opt.print_freq == 0:
            # vis.plot('loss', loss_meter.value()[0])
            print("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f" % (epoch + 1, EPOCH_NUM, i + 1, BATCH_NUM, loss.data[0]))
            
            plt.plot(train_Loss_record, color="red")
            if i == 36037:
                plt.savefig("Results.png")
            plt.pause(0.001)
            plt.clf()
        
        torch.save(my_model, MODEL_PATH)
            lr = lr * opt.lr_decay
'''
    # validate and visualize
    val_cm, val_accuracy = val(my_model, val_loader)
    
    vis.plot('val_accuracy', val_accuracy)
    vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
    epoch=epoch, loss=loss_meter.value()[0], val_cm=str(val_cm.value()), train_cm=str(confusion_matrix.value()),
    lr=LR))
    
    # update learning rate
    if loss_meter.value()[0] > previous_loss:
    lr = lr * opt.lr_decay
    # 第二种降低学习率的方法:不会有moment等信息的丢失
    for param_group in optimizer.param_groups:
    param_group['lr'] = lr
    
    previous_loss = loss_meter.value()[0]
    '''
