TEST_PATH = "final_data/test/input/"
TEST_SUFFIX = ".jpg"
TXT_PATH = "concat.txt"

for i in range(180):
    for j in range(576):
        line = TEST_PATH + str(i) + "_" + str(j) + TEST_SUFFIX
        # write line in txt
        f = open(TXT_PATH, 'a')
        f.write(line + "  " + line + "\n")
f.close()
