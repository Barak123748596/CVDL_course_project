import numpy as np
from PIL import Image

def isgreen(x):
    # if x[0]<80 and x[1]>115 and x[2]<80:
    if x[1]>x[0]+30 and x[1]>x[2]+30:
        return 1
    else:
        return 0


def promatrix_prime(im):
    size = im.shape
    new_pic = np.zeros(size)
    m=size[0]
    n=size[1]
    for i in range(m):
        for j in range(n):
            if isgreen(im[i][j])==1:
                new_pic[i][j] = [0,255,0]
                im[i][j] = [255,0,0]
            else:
                new_pic[i][j] = [255,255,255]
            # print('i:',i,'j:',j,'new:', new_pic[i][j],'\n')
    newim=Image.fromarray(im.astype(np.uint8))
    newim.show()#查看新生成的图像
    newim.save('new.png')

    trans_im = Image.fromarray(new_pic.astype(np.uint8))
    # trans_im.show()  # 查看新生成的图像
    trans_im.save('trans.png')
    return new_pic

# if __name__ == "main":
im0=Image.open('北京市.jpg')
im=np.array(im0)
immatrix=promatrix_prime(im)
# np.save("array.npy",immatrix)