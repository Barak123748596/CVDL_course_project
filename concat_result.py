from utils.config import opt
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
stride = 160
seg = 320
L_NUM = 31
BATCH_NUM = 31 * 31
TEST_CITY_NAME = ["bellingham", "bloomington", "innsbruck", "sfo", "tyrol-e"]
TEST_RESULT_PATH = "final_data/test/output/"

# 0: Tesla K20C   1: Quadro 600
print(torch.cuda.is_available())

my_model = torch.load(MODEL_PATH).cuda()
print("model from load")
# print(my_model)

criterion = torch.nn.NLLLoss()

avg_img = np.zeros([5120, 5120, 3])
avg_num = np.ones([5120, 5120], np.int)

for i in range(L_NUM):
    for j in range(L_NUM):
        avg_num[i*stride:i*stride+seg, j*stride:j*stride+seg] += 1

avg_prob = np.zeros([5120, 5120])  # Final submit prob

# get result
for i, (images, labels) in enumerate(test_loader):
    city_id = i // (36*BATCH_NUM)
    big_id = (i % (36*BATCH_NUM)) // BATCH_NUM
    tmp_row = (i % BATCH_NUM) // L_NUM
    tmp_col = (i % BATCH_NUM) % L_NUM
    
    # img_show = np.squeeze(images.numpy(), 0)
    # img_show = img_show.transpose((1, 2, 0))
    # cv2.imshow("tmp_win1", img_show)
    # cv2.waitKey(0)
    
    if torch.cuda.is_available():
        images = images.cuda()

    img_show = np.squeeze(images.cpu().detach().numpy(), 0)
    img_show = img_show.transpose((1, 2, 0))
    avg_img[tmp_row * stride:tmp_row * stride + seg, tmp_col * stride:tmp_col * stride + seg, :] = img_show

    # Forward
    outputs = my_model(images)
    img_out = np.squeeze((outputs.detach()).cpu().numpy(), 0)
    img_out = img_out.transpose((1, 2, 0))
    prob = np.exp(img_out[:, :, 1])

    avg_prob[tmp_row*stride:tmp_row*stride+seg, tmp_col*stride:tmp_col*stride+seg] += prob
    
    # Show result for small img
    # prob[prob < 0.5] = 0
    # prob[prob >= 0.5] = 1
    # img_print = 255.0 * prob
    # cv2.imshow("tmp_win2", img_print)
    # cv2.waitKey(0)
    
    if (i+1) % BATCH_NUM == 0:
        avg_prob = avg_prob / avg_num
        avg_prob[avg_prob < 0.5] = 0
        avg_prob[avg_prob >= 0.5] = 1
        # img_submit = 255 * avg_prob
        avg_prob = avg_prob[60:5060, 60:5060]
        img_submit = np.array(img_show[60:5060, 60:5060, :])
        img_submit[avg_prob == 1, 0:2] = 40
        cv2.imwrite(TEST_RESULT_PATH + TEST_CITY_NAME[city_id] + str(big_id + 1) + ".jpg", img_submit)
        print("Saving...")
        avg_prob = np.zeros_like(avg_prob)
