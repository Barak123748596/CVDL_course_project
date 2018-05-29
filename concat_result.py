from config import opt
import torch
from LoadData import train_loader, test_loader
from torch.autograd import Variable
from torch import nn
from U_Net import U_Net
import numpy as np
import cv2

EPOCH_NUM = 1
MODEL_PATH = "models/V1.1/U_Net.pkl"
N_CHANNEL = 3
N_CLASS = 2
BATCH_NUM = 24 * 24
TEST_CITY_NAME = ["bellingham", "bloomington", "innsbruck", "sfo", "tyrol-e"]
TEST_RESULT_PATH = "final_data/test/output/"

# 0: Tesla K20C   1: Quadro 600
print(torch.cuda.is_available())
print(torch.cuda.device_count())

my_model = torch.load(MODEL_PATH, map_location='cpu')
print("model from load")
print(my_model)

criterion = torch.nn.NLLLoss()

avg_num = np.ones([5000, 5000], np.int)
avg_num[200:4800, 200:4800] = 4
avg_num[0, 200:4800] = 2
avg_num[200:4800, 0] = 2
avg_num[4999, 200:4800] = 2
avg_num[200:4800, 4999] = 2

avg_prob = np.zeros([5000, 5000])  # Final submit prob

# get result
for i, (images, labels) in enumerate(test_loader):
    city_id = i // (36*24*24)
    big_id = (i % (36*24*24)) // (24*24)
    tmp_row = (i % (24*24)) // 24
    tmp_col = (i % (24*24)) % 24

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

    avg_prob[tmp_row*400:(tmp_row+1)*400, tmp_col*400:(tmp_col+1)*400] += prob
    # Show result for small img
    prob[prob < 0.5] = 0
    prob[prob >= 0.5] = 1
    img_print = 255.0 * prob
    cv2.imshow("tmp_win2", img_print)
    cv2.waitKey(0)

    if (i+1) % BATCH_NUM == 0:
        avg_prob = avg_prob / avg_num
        avg_prob[avg_prob < 0.5] = 0
        avg_prob[avg_prob >= 0.5] = 1
        img_submit = 255 * avg_prob
        cv2.imwrite(TEST_RESULT_PATH + TEST_CITY_NAME[city_id] + str(big_id + 1) + ".tif", img_submit)
        avg_prob = 0
