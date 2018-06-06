import numpy as np
import random

seed = 2147483647

random.seed(seed)
np.random.seed(seed)
print(np.random.random())
print(random.random())
print(random.random())

random.seed(seed)
np.random.seed(seed)
print(np.random.random())
print(random.random())
print(random.random())