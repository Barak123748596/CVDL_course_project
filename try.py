import importlib
import torchvision
from PIL import Image
import numpy as np
import torch

a = Image.open("lena.jpg").convert("1")
print(np.array(a)[3,3])