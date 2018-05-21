import torch
from LoadData import train_loader, val_loader
from torch.autograd import Variable
from torch import nn
from torchvision import models
import os
from U_Net import U_Net
import numpy as np

EPOCH_NUM = 10
MODEL_PATH = "models/V1.0/U_Net.pkl"
N_CHANNEL = 3
N_CLASS = 2
LR = 2e-5
BATCH_NUM = 36038

# if os.path.exists(r'transfer_resnet18.pkl'):
#     my_model = torch.load('transfer_resnet18.pkl').cuda()
#     print("model from load")
# else:
#     my_model = models.resnet18(pretrained=True).cuda()
#     torch.save(my_model, 'transfer_resnet18.pkl')
#     print("model build")

my_model = U_Net(N_CHANNEL, N_CLASS).cuda()
print(my_model)

# criterion = torch.nn.CrossEntropyLoss().cuda()
criterion = torch.nn.NLLLoss().cuda()
optimizer = torch.optim.Adam(my_model.parameters(), lr=LR)

for epoch in range(EPOCH_NUM):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.cuda())
        labels = Variable(labels.long().cuda())
        labels = labels.squeeze(1)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = my_model(images)
        # print(np.shape(outputs))
        # print(np.shape(labels))
        loss = criterion(outputs, labels)
        # print(loss)
        loss.backward()
        optimizer.step()

        # if (i + 1) % 100 == 0:
        print("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f" % (epoch + 1, EPOCH_NUM, i + 1, BATCH_NUM, loss.data[0]))
    torch.save(my_model, MODEL_PATH)
