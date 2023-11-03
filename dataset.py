from torch.utils.data import Dataset
from transformers import AutoTokenizer
import numpy as np
import torch

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
        # Read raw data into dataset
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
                        tokenized_sentence = tokenizer(
                            sentence,
                            padding=True,
                            truncation=True,
                            max_length=512,
                            return_tensors='pt'
                        )
                        self.sentences.append(tokenized_sentence['input_ids'])
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

def collate_fn(data):
    max_length = 512
    batch_sentence = []
    batch_label = []
    for sentence, label in data:
        collated_sentence = np.pad(np.array(sentence), ((0, max_length - len(sentence))))
        batch_sentence.append(torch.Tensor(collated_sentence))
        
        batch_label.append(torch.Tensor(label))
    print(len(batch_sentence))
    print(len(batch_label))
    return torch.Tensor(batch_sentence), torch.Tensor(batch_label)
        