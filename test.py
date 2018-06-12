# from config import opt
import torch
from LoadData import train_loader, test_loader, beijing_loader
from torch.autograd import Variable
from torch import nn
from torchvision import models
from U_Net import U_Net
import numpy as np
import cv2

EPOCH_NUM = 1
MODEL_PATH = "models/V1.4/U_Net.pkl"
N_CHANNEL = 3
N_CLASS = 2
BATCH_NUM = 961

# 0: Tesla K20C   1: Quadro 600
print(torch.cuda.is_available())
print(torch.cuda.device_count())

my_model = torch.load(MODEL_PATH)
print("model from load")
print(my_model)

criterion = torch.nn.NLLLoss()

# test
for epoch in range(EPOCH_NUM):
    for i, (images, labels) in enumerate(beijing_loader):
        images = Variable(images)

        img_show = np.squeeze(images.numpy(), 0)
        img_show = img_show.transpose((1, 2, 0))
        # print(np.shape(img_show))
        cv2.imshow("tmp_win1", img_show)
        cv2.imwrite(str(epoch)+"_"+str(i)+"_0.jpg", img_show*255)
        cv2.waitKey(0)

        if torch.cuda.is_available():
            images = images.cuda()

        # Forward
        outputs = my_model(images)
        img_out = np.squeeze((outputs.detach()).cpu().numpy(), 0)
        img_out = img_out.transpose((1, 2, 0))

        cv2.imshow("tmp_win2", -img_out[:, :, 1])
        cv2.waitKey(0)

        prob = np.exp(img_out[:, :, 1])
        cv2.imwrite(str(epoch)+"_"+str(i)+"_1.jpg", prob*255)
        prob[prob < 0.6] = 0
        prob[prob >= 0.6] = 1.0
        cv2.imshow("tmp_win3", prob)
        cv2.waitKey(0)
