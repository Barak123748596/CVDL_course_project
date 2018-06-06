# Train txt
TRAIN_PATH_INPUT = "final_data/train/input/"
TRAIN_PATH_OUTPUT = "final_data/train/output/"
TRAIN_SUFFIX = ".jpg"
TRAIN_TXT_PATH = "train.txt"

f = open(TRAIN_TXT_PATH, 'a')
for i in range(5):
    for j in range(i*36, i*36+31):
        for k in range(961):
            line_in = TRAIN_PATH_INPUT + str(j) + "_" + str(k) + TRAIN_SUFFIX
            line_out = TRAIN_PATH_OUTPUT + str(j) + "_" + str(k) + TRAIN_SUFFIX
            # write line in txt
            f.write(line_in + "  " + line_out + "\n")
f.close()

# Val txt
VAL_PATH_INPUT = "final_data/train/input/"
VAL_PATH_OUTPUT = "final_data/train/output/"
VAL_SUFFIX = ".jpg"
VAL_TXT_PATH = "val.txt"

f = open(VAL_TXT_PATH, 'a')
for i in range(5):
    for j in range(i*36+31, (i+1)*36):
        for k in range(961):
            line_in = VAL_PATH_INPUT + str(j) + "_" + str(k) + VAL_SUFFIX
            line_out = VAL_PATH_OUTPUT + str(j) + "_" + str(k) + VAL_SUFFIX
            # write line in txt
            f.write(line_in + "  " + line_out + "\n")
f.close()

# Test txt
TEST_PATH = "final_data/test/input/"
TEST_SUFFIX = ".jpg"
TEST_TXT_PATH = "test.txt"

f = open(TEST_TXT_PATH, 'a')
for i in range(180):
    for j in range(961):
        line = TEST_PATH + str(i) + "_" + str(j) + TEST_SUFFIX
        # write line in txt
        f.write(line + "  " + line + "\n")
f.close()
