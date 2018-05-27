import torch
import numpy as np
import cv2

train_input_path = 'raw_data/train/input/'
train_output_path = 'raw_data/train/output/'
train_input_save = 'final_data/train/input/'
train_output_save = 'final_data/train/output/'

image_rows = 5000
image_cols = 5000

x_stride = 150
y_stride = 150

x_slice_num = image_cols // x_stride
y_slice_num = image_rows // y_stride  # 33

x_seg = 400
y_seg = 400

DATA_NUM = 180
for i in range(DATA_NUM):
    print(i+1, "big image...")
    img_path = train_input_path + "austin" + str(i+1) + ".tif"
    ans_path = train_output_path + "austin" + str(i+1) + ".tif"

    img_array = cv2.imread(img_path)
    ans_array = cv2.imread(ans_path, 0).reshape(5000, 5000, 1)
    print(ans_array.shape)

    batch_small_img = np.zeros([x_slice_num*y_slice_num, x_seg, y_seg, 3])
    batch_small_ans = np.zeros([x_slice_num*y_slice_num, x_seg, y_seg, 1])
    l = 0
    for j in range(y_slice_num-2):
        for k in range(x_slice_num-2):
            y_start = j * y_stride
            y_end = y_start + y_seg
            x_start = k * x_stride
            x_end = x_start + x_seg
            print(x_start, x_end)
            batch_small_img[l, :, :, :] = img_array[y_start:y_end, x_start:x_end, :]
            batch_small_ans[l, :, :, :] = ans_array[y_start:y_end, x_start:x_end, :]
            l += 1

    for m in range(l):
        print("saving", m, "small img...")
        tmp_img = batch_small_img[m]
        tmp_ans = batch_small_ans[m]
        save_path = train_input_save + str(i) + "_" + str(m) + ".png"
        cv2.imwrite(save_path, tmp_img)
        save_path = train_output_save + str(i) + "_" + str(m) + ".png"
        cv2.imwrite(save_path, tmp_ans)
