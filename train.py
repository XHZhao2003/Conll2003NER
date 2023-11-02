from torch.utils.data import DataLoader
from torch.optim import optimizer
from dataset import ConllDataset

train_dataset = ConllDataset("data/test.txt")
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)


