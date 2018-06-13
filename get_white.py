# 负责把mask图片中的楼用bounding box框起来（不严格）
import numpy as np
import cv2

# product函数将矩阵一行中若干段相连的白点提取出来

PATH = "final_data/train/output/0_15.jpg"

def product(k,x):
    l=len(x)
    s=[]
    i=0
    while i<l:
        if x[i]==1:
            t=i
            p=[k,k]
            p.append(i)
            i+=1
            while i<l and x[i]==1:
                i+=1
            i-=1
            p.append(i)
            p.append(t)
            p.append(i)
            p.append(t)
            p.append(i)
            s.append(p)
        i+=1
    return s
##modify函数将一个矩形条添加到已有的矩形列表中，将与之相连的合并为一个
def modify(s,p):
    n=len(s)
    k=p[0]
    a1=p[2]
    a2=p[3]
    l=[]
    t=n
    ##找出与新的矩形p相连的矩形，存到l中
    for i in range(n):
        if s[i][1]==k-1:
            if not (s[i][4]>a2 or s[i][5]<a1):
                l.append(s[i])
                t=min(t,i)
        if s[i][1]==k:
            if not (s[i][6]>a2 or s[i][7]<a1):
                l.append(s[i])
                t=min(t,i)
    if len(l)>0:
        m=len(l)
        b1=p[0]
        b2=p[1]
        c1=p[2]
        c2=p[3]
        d1=p[4]
        d2=p[5]
        e1=n
        e2=0
        for j in range(m):
            b1=min(b1,l[j][0])
            b2=max(b2,l[j][1])
            c1=min(c1,l[j][2])
            c2=max(c2,l[j][3])
            if l[j][1]==k:
                d1=min(d1,l[j][4])
                d2=max(d2,l[j][5])
                e1=min(e1,l[j][6])
                e2=max(e2,l[j][7])
            else:
                e1=min(e1,l[j][4])
                e2=max(e2,l[j][5])
            s.remove(l[j])
        x=[b1,b2,c1,c2,d1,d2,e1,e2]
        s.insert(t,x)#将新的矩阵放到原有列中最初参与合并的位置
    else:
        s.append(p)
    return s
##change函数将输出的列表转化为(n,x,y,w,h)的形式
def change(s):
    n=len(s)
    for i in range(n):
        p=[i+1,s[i][0]+1,s[i][2]+1,s[i][1]-s[i][0]+1,s[i][3]-s[i][2]+1]
        s[i]=p
    return s


if __name__ == '__main__':
    # picture表示代表图片的01矩阵，1代表白色
    picture = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
    # print(picture[0])
    picture = np.around(np.array(picture/255.0))
    m = np.shape(picture)[0]  # 矩阵行数
    n = np.shape(picture)[1]  # 矩阵列数
    cv2.imshow("tmpwin", picture)
    cv2.waitKey(0)

    s = []

    print(picture[0])  # Debug

    for i in range(0, m):
        # print(i, "of", m)
        l = product(i, picture[i])
        print(i, ":", l)
        k = len(l)
        for j in range(k):
            s = modify(s, l[j])

    print(s)
    s = change(s)
    picture = cv2.imread(PATH, cv2.IMREAD_COLOR)
    print("how many:", len(s))
    for term in s:
        cv2.rectangle(picture, (term[2], term[1]), (term[2]+term[4]-1, term[1]+term[3]-1), (0, 255, 0), 1)
    cv2.imshow("tmpwin", picture)
    cv2.waitKey(0)
