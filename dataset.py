from torch.utils.data import Dataset
from torch.nn import functional as F
from transformers import AutoTokenizer
import numpy as np
import torch

# example: 
#   onehot(list[0, 2, 3], 5)
#   > Tensor[[1, 0, 0, 0, 0],[0, 0, 1, 0, 0], [0, 0, 0, 1, 0]]
def one_hot(label_list, num_class=9):
    label = []
    for x in label_list:
        label.append([x])
    return torch.zeros(len(label_list), num_class).scatter(1, torch.Tensor(label).long(), 1)

# input: a batch of (raw sentence, label)
# output: input_ids, token_type_ids, attention_masks, one_hot_labels (after padding)
# shape for bert_input: batch * max_length
# shape for labels: batch * max_length * num_classes
def collate_fn(data):
    sentences, labels = [], []
    max_length = 512
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    for sentence, _ in data:
        sentences.append(sentence)
    
    tokenized_sentence = tokenizer(
        sentences,
        padding='max_length',
        truncation = True,
        max_length = max_length,
        return_tensors='pt'
    )
    
    
    for _, label in data:
        # [CLS, tokens, PAD, ...]
        pad_label = [0] + label + [0] * (max_length - 1 - len(label))
        
        one_hot_label = one_hot(pad_label)
        labels.append(torch.unsqueeze(one_hot_label, 0))
    one_hot_labels = torch.cat([x for x in labels], 0)
    
    return tokenized_sentence['input_ids'], tokenized_sentence['token_type_ids'], tokenized_sentence['attention_mask'], one_hot_labels

class ConllDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.max_length = 512
        self.data_path = data_path
        self.sentences = []
        self.labels = []
        self.label2id = {
            'O': 0,
            'B-PER': 1,
            'I-PER': 2,
            'B-ORG': 3,
            'I-ORG': 4,
            'B-LOC': 5,
            'I-LOC': 6,
            'B-MISC': 7,
            'I-MISC': 8}
        self.ReadRawData()

    def ReadRawData(self):
        # Read raw data 
        # save raw sentence and labels
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        doc_start = "-DOCSTART- -X- -X- O\n"
        with open(self.data_path, mode='r') as data_file:
            lines = data_file.readlines()
            sentence = ""
            label = []
            for lineString in lines:
                line = lineString.split()
                if lineString == doc_start:
                    continue
                if len(line) == 0:
                    if len(sentence) > 0:
                        self.sentences.append(sentence)
                        self.labels.append(label)
                        sentence = ""
                        label = []
                    continue
                if sentence != "":
                    sentence += ' '
                sentence += line[0]
                label.append(self.label2id[line[3]])

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        return self.sentences[item], self.labels[item]

        