# # from model import BertModelNer
# # from dataset import ConllDataset
# # from torch.utils.data import DataLoader

# # dataset = ConllDataset("data/test.txt")
# # loader = DataLoader(dataset, batch_size=16, shuffle=False)
# # model = BertModelNer(label_num=9)

# # for batch in loader:
# #     print(batch)
    
# #     break

# from transformers import AutoTokenizer

# sentence = "The cat ate the fat rat"
# tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased',force_download=True)
# tokenized_sentence = tokenizer(sentence, 
#                                padding=True, 
#                                truncation=True,
#                                max_length=512, 
#                                return_tensors='pt')
# print(tokenized_sentence)

a = [(1, 2)]
for b, c in a:
    print(b, c)