import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import random

PATH_FOLDER = ""

# def default_img_loader(path):
#     return Image.open(path).convert("RGB")
#
# def default_label_loader(path):
#     return Image.open(path).convert("1")


def default_img_loader(path):
    # img = cv2.imread(path)
    # print(np.shape(img))
    # return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return Image.open(path).convert('RGB')


def default_label_loader(path):
    # img = cv2.imread(path)
    # print(np.shape(img))
    # return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    # return Image.fromarray(cv2.imread(path, 0).reshape(400, 400, 1))
    return Image.open(path).convert('L')


class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None,
                 img_loader=default_img_loader, label_loader=default_label_loader, add_wrd=PATH_FOLDER):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((add_wrd+words[0], add_wrd+words[1]))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.img_loader = img_loader
        self.label_loader = label_loader
    
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.img_loader(fn)
        label = self.label_loader(label)
        if self.transform is not None:
            # Same Seed
            seed = np.random.randint(2147483647)
            random.seed(seed)  # apply this seed to img transforms
            img = self.transform(img)
            random.seed(seed)  # apply this seed to img transforms
            label = self.transform(label)
        return img, label
    
    def __len__(self):
        return len(self.imgs)


train_data = MyDataset(txt="train.txt",
                       transform=transforms.Compose([transforms.RandomRotation(180),
                                                     transforms.RandomHorizontalFlip(),
                                                     transforms.RandomVerticalFlip(),
                                                     transforms.ColorJitter(brightness=.3,
                                                                            contrast=.3,
                                                                            saturation=.3),
                                                     transforms.ToTensor()
                                                     ]))
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)

val_data = MyDataset(txt="val.txt",
                     transform=transforms.Compose([transforms.RandomRotation(180),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.RandomVerticalFlip(),
                                                   transforms.ColorJitter(brightness=.3,
                                                                          contrast=.3,
                                                                          saturation=.3),
                                                   transforms.ToTensor()
                                                   ]))
val_loader = DataLoader(val_data, batch_size=4, shuffle=True)

test_data = MyDataset(txt='concat.txt',
                      transform=transforms.Compose([transforms.ToTensor()]))
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

print("load complete!")
