import numpy as np
import cv2

train_input_path = 'raw_data/train/input_try/'
train_output_path = 'raw_data/train/output_try/'
train_input_save = 'final_data/train/input/'
train_output_save = 'final_data/train/output/'

test_input_path = 'raw_data/test/input_try/'
test_output_path = 'raw_data/test/output_try/'
test_input_save = 'final_data/test/input/'
test_output_save = 'final_data/test/output/'

image_rows = 5000
image_cols = 5000

x_stride = 150
y_stride = 150

x_seg = 300
y_seg = 300

x_slice_num = 1 + ((image_cols - x_seg) // x_stride)
y_slice_num = 1 + ((image_rows - y_seg) // y_stride)  # 32

print(x_slice_num, y_slice_num)

CITY_NUM = 5
TRAIN_CITY_NAME = ["austin", "chicago", "kitsap", "tyrol-w", "vienna"]
TEST_CITY_NAME = ["bellingham", "bloomington", "innsbruck", "sfo", "tyrol-e"]
EACH_CLASS_NUM = 36

TOTAL_RATIO = 0
'''
    for i in range(CITY_NUM):
    for j in range(EACH_CLASS_NUM):
    tmp_id = i * EACH_CLASS_NUM + j
    print(tmp_id+1, "big image...")
    img_path = train_input_path + TRAIN_CITY_NAME[i] + str(j + 1) + ".tif"
    ans_path = train_output_path + TRAIN_CITY_NAME[i] + str(j + 1) + ".tif"
    
    img_array = cv2.imread(img_path, cv2.IMREAD_COLOR)
    ans_array = cv2.imread(ans_path, cv2.IMREAD_GRAYSCALE).reshape(5000, 5000, 1)
    # print("positive:", np.sum(ans_array))
    # TOTAL_RATIO += np.sum(ans_array) / (5000 * 5000 * 255)
    # AVG RATIO: 0.1578
    
    batch_small_img = np.zeros([x_slice_num * y_slice_num, x_seg, y_seg, 3])
    batch_small_ans = np.zeros([x_slice_num * y_slice_num, x_seg, y_seg, 1])
    m = 0
    
    for k in range(y_slice_num):
    for l in range(x_slice_num):
    y_start = k * y_stride
    y_end = y_start + y_seg
    x_start = l * x_stride
    x_end = x_start + x_seg
    # print(x_start, x_end)
    batch_small_img[m, :, :, :] = img_array[y_start:y_end, x_start:x_end, :]
    batch_small_ans[m, :, :, :] = ans_array[y_start:y_end, x_start:x_end, :]
    m += 1
    
    for n in range(m):
    print("saving", n, "small img...")
    tmp_img = batch_small_img[n]
    tmp_ans = batch_small_ans[n]
    save_path = train_input_save + str(tmp_id) + "_" + str(n) + ".jpg"
    cv2.imwrite(save_path, tmp_img)
    save_path = train_output_save + str(tmp_id) + "_" + str(n) + ".jpg"
    cv2.imwrite(save_path, tmp_ans)
    '''

for i in range(CITY_NUM):
    for j in range(EACH_CLASS_NUM):
        tmp_id = i * EACH_CLASS_NUM + j
        print(tmp_id + 1, "big image...")
        img_path = test_input_path + TEST_CITY_NAME[i] + str(j + 1) + ".tif"
        ans_path = test_output_path + TEST_CITY_NAME[i] + str(j + 1) + ".tif"
        
        img_array = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # ans_array = cv2.imread(ans_path, cv2.IMREAD_GRAYSCALE).reshape(5000, 5000, 1)
        # print(ans_array.shape)
        
        batch_small_img = np.zeros([x_slice_num * y_slice_num, x_seg, y_seg, 3])
        # batch_small_ans = np.zeros([x_slice_num * y_slice_num, x_seg, y_seg, 1])
        m = 0
        
        for k in range(y_slice_num):
            for l in range(x_slice_num):
                y_start = k * y_stride
                y_end = y_start + y_seg
                x_start = l * x_stride
                x_end = x_start + x_seg
                # print(x_start, x_end)
                batch_small_img[m, :, :, :] = img_array[y_start:y_end, x_start:x_end, :]
                # batch_small_ans[m, :, :, :] = ans_array[y_start:y_end, x_start:x_end, :]
                m += 1
    
        for n in range(m):
            print("saving", n, "small img...")
            tmp_img = batch_small_img[n]
            # tmp_ans = batch_small_ans[n]
            save_path = test_input_save + str(tmp_id) + "_" + str(n) + ".jpg"
            cv2.imwrite(save_path, tmp_img)
# save_path = train_output_save + str(tmp_id) + "_" + str(n) + ".jpg"
# cv2.imwrite(save_path, tmp_ans)
