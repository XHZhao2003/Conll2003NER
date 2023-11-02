import os

import torch
from transformers import AutoTokenizer, BertModel
from torch import nn, functional

class BertModelNer(nn.Module):
    def __init__(self, label_num=9):
        super().__init__()
        self.bertModel = BertModel.from_pretrained('bert-base-uncased')

        self.classifier = nn.Sequential(
            nn.Linear(in_features=768, out_features=768),
            nn.ReLU(),
            nn.Linear(in_features=768, out_features=label_num),
            nn.Softmax(dim=2)
        )

    def forward(self, input_ids):
        last_hidden_state = self.bertModel(input_ids).last_hidden_state
        print(type(last_hidden_state))
        print(last_hidden_state.shape)
        return self.classifier(last_hidden_state)

    def SaveModel(self):
        os.makedirs("trained_model", exist_ok=True)
        torch.save(self, os.path.join("trained_model/model.pt"))

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = BertModelNer()
#
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = model(inputs['input_ids'])
#
# print(outputs)
# print(outputs.shape)
