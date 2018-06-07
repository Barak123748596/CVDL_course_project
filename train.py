from utils.config import opt
import torch
from LoadData import train_loader, val_loader
from torch.autograd import Variable
from torch import nn
from torchvision import models
import os
from U_Net import U_Net, U_Net_pile
import numpy as np
from utils.visualize import Visualizer
# from torchnet import meter
from matplotlib import pyplot as plt
import random
import cv2

EPOCH_NUM = 5
MODEL_PATH = "models/V1.3/U_Net.pkl"
N_CHANNEL = 3
N_CLASS = 2
LR = 1e-6
BATCH_NUM = 150*961 / train_loader.batch_size

def adjust_learning_rate(optimizer, decay_rate=0.5):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate


# def val(model, dataloader):
#     '''
#         计算模型在验证集上的准确率等信息
#         :param model:
#         :param dataloader:
#         :return:
#         '''
#     model.eval()
#     if torch.cuda.is_available():
#         model = model.cuda()
#
#     confusion_matrix = meter.ConfusionMeter(2)
#     for ii, data in enumerate(dataloader):
#         input, label = data
#         val_input = Variable(input, volatile=True)
#         val_label = Variable(label.long(), volatile=True)
#
#         if torch.cuda.is_available():
#             val_input = val_input.cuda()
#             val_label = val_label.cuda()
#
#         score = model(val_input)
#         # print(np.shape(val_label.cpu().detach().numpy()))
#         # print(np.shape(score.cpu().detach().numpy()))
#
#         label_flat = (val_label.view(-1, 1)).squeeze(1)
#         score_flat = score.view(-1, 2)
#
#         # print(np.shape(label_flat.cpu().detach().numpy()))
#         # print(np.shape(score_flat.cpu().detach().numpy()))
#
#         confusion_matrix.add(score_flat.cpu().detach(), label_flat.cpu().detach())
#
#     # 把模型恢复为训练模式
#     model.train()
#
#     cm_value = confusion_matrix.value()
#     accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / \
#         (cm_value.sum())
#     return confusion_matrix, accuracy


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

    for ii, data in enumerate(dataloader):
        input, label = data
        val_input = Variable(input, volatile=True)
        val_label = Variable(label.long(), volatile=True)

        criterion = torch.nn.NLLLoss(weight=torch.FloatTensor([0.45, 1.55]))

        if torch.cuda.is_available():
            val_input = val_input.cuda()
            val_label = val_label.cuda().squeeze(1)
            criterion = criterion.cuda()
        val_outputs = model(val_input)
        _, val_indices = torch.max(val_outputs, 1)
        val_loss = criterion(val_outputs, val_label)
        break

    # 把模型恢复为训练模式
    model.train()
    return val_loss


# vis = Visualizer(opt.env)
if os.path.exists(MODEL_PATH):
    my_model = torch.load(MODEL_PATH)
    print("model from load.")
else:
    # my_model = U_Net(N_CHANNEL, N_CLASS)
    my_model = U_Net_pile(N_CHANNEL, N_CLASS, pile=2)
    print("A new model.")
# print(my_model)

criterion = torch.nn.NLLLoss(weight=torch.FloatTensor([0.45, 1.55]))
optimizer = torch.optim.Adam(my_model.parameters(), lr=LR)

if torch.cuda.is_available():
    my_model = my_model.cuda(device=0)
    criterion.cuda(device=0)

# 统计指标：平滑处理之后的损失，还有混淆矩阵
# loss_meter = meter.AverageValueMeter()
# confusion_matrix = meter.ConfusionMeter(2)
# previous_loss = 1e100

train_Loss_record = []
val_Loss_record = []

tmp_loss = 0
tmp_val_loss = 0

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
            images = images.cuda(device=0)
            labels = labels.cuda(device=0)
        labels = labels.squeeze(1)

        # # For show augmentation
        # img_show = images.cpu().detach().numpy()
        # img_show = np.transpose(img_show, (0, 2, 3, 1))
        # cv2.imshow("tmpwin", img_show[0])
        # cv2.waitKey(0)

        # Forward + Backward + Optimize
        optimizer.zero_grad()

        # 设置高斯噪声
        if random.random() < 1:
            images = torch.add(images, torch.Tensor(np.random.normal(0, 0.04, np.array(images).size)).view(np.array(images).shape).cuda())

        outputs = my_model(images)
        loss = criterion(outputs, labels)
        tmp_loss += 0.1 * loss.cpu().detach().numpy()
        loss.backward()
        optimizer.step()

        _, indices = torch.max(outputs, 1)
        correct = (indices == labels).sum()
        Intersect = (indices * labels == 1).sum()
        Union = indices.sum() + labels.sum() - Intersect
        IoU = 0
        if Union > 0:
            IoU = float(Intersect) / float(Union + .01)

        val_loss = val(my_model, val_loader)
        tmp_val_loss += 0.1 * val_loss.cpu().detach().numpy()

        if (i+1) % 10 == 0:
            train_Loss_record.append(tmp_loss)
            val_Loss_record.append(tmp_val_loss)
            tmp_loss = 0
            tmp_val_loss = 0

            plt.plot(train_Loss_record, color="red")
            plt.plot(val_Loss_record, color="blue")
            plt.clf()

        if i % opt.print_freq == 0:
            # vis.plot('loss', loss_meter.value()[0])
            print("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, Acc: %.2f, IoU: %.2f"
                  % (epoch + 1, EPOCH_NUM, i + 1, BATCH_NUM, loss.item(),
                     100.0 * correct / (320*320*train_loader.batch_size), 100.0 * IoU))
            print("     val loss: %.4f" % val_loss)
            plt.savefig("Results.png")

    torch.save(my_model, MODEL_PATH)
    adjust_learning_rate(optimizer=optimizer, decay_rate=opt.lr_decay)
    '''
    # validate and visualize
    val_cm, val_accuracy = val(my_model, val_loader)
    
    vis.plot('val_accuracy', val_accuracy)
    vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
    epoch=epoch, loss=loss_meter.value()[0], val_cm=str(val_cm.value()), train_cm=str(confusion_matrix.value()),
    lr=LR))
    
    # update learning rate
    if loss_meter.value()[0] > previous_loss:
        LR = LR * opt.lr_decay
    # 第二种降低学习率的方法:不会有moment等信息的丢失
    for param_group in optimizer.param_groups:
        param_group['lr'] = LR
    
    previous_loss = loss_meter.value()[0]
    '''
