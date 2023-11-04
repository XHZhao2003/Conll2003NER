import torch
from torch.utils.data import DataLoader
from torch import optim, nn
from tqdm import tqdm

from dataset import ConllDataset, collate_fn
from model import BertModelNer

batch_size = 8
epoch_num = 5
lr_rate = 1e-3
label_num = 9



train_dataset = ConllDataset("data/test.txt")
print(train_dataset.__len__())
train_dataloader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              collate_fn=collate_fn)

model = BertModelNer(label_num=label_num)
optimizer = optim.Adam(model.parameters(), lr=lr_rate)
loss_fn = nn.CrossEntropyLoss()

# para_num = 0
# for p in model.parameters():
#     para_num += p.numel()   
# print("Total number of parameters is %d" %(para_num))

for epoch in tqdm(range(1, epoch_num + 1), ncols=80):
    for input_ids, token_type_ids, attention_mask, label in tqdm(train_dataloader, ncols=80):
        optimizer.zero_grad()
        logits = model(input_ids, token_type_ids, attention_mask)
        
    
    

