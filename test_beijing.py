from utils.config import opt
import torch
from LoadData import beijing_loader
from torch.autograd import Variable
from torch import nn
from torchvision import models
from U_Net import U_Net
import numpy as np
import cv2

EPOCH_NUM = 1
MODEL_PATH = "models/V1.3/U_Net.pkl"
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
        cv2.waitKey(0)

        if torch.cuda.is_available():
            images = images.cuda()

        # Forward
        outputs = my_model(images)
        img_out = np.squeeze((outputs.detach()).cpu().numpy(), 0)
        img_out = img_out.transpose((1, 2, 0))
        prob = np.exp(img_out[:, :, 1])
        prob[prob < 0.5] = 0
        prob[prob >= 0.5] = 1
        img_print = 255.0 * prob


        # print(img_print)
        cv2.imshow("tmp_win2", img_print)
        cv2.waitKey(0)

        if i % opt.print_freq == 0:
            print("Epoch [%d/%d], Iter [%d/%d]..." % (epoch + 1, EPOCH_NUM, i + 1, BATCH_NUM))
