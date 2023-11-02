from torch.utils.data import Dataset

class ConllDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
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
        return [self.sentences[item], self.labels[item]]


        
    


