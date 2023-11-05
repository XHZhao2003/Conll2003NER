from torch.nn import CrossEntropyLoss
from torch import Tensor as t
from tqdm import tqdm
import time

for i in tqdm(range(10), ncols=50, total=10):
    time.sleep(2)