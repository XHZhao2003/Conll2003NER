from transformers import AutoTokenizer

sentences = ["I am in Paris"]
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
print(tokenizer(sentences, padding='max_length', truncation=True, max_length=512, return_tensors='pt'))