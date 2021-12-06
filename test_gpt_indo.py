# from transformers import GPT2Tokenizer, GPT2Model
# # 

# # model_name = 'flax-community/gpt2-small-indonesian'
# model_name = 'cahya/gpt2-small-indonesian-522M'

# tokenizer = GPT2Tokenizer.from_pretrained(model_name, force_download=True)
# model = GPT2Model.from_pretrained(model_name, force_download=True)
# text = "Ubah dengan teks apa saja."
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)
# print(output)
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer

tokenizer = TreebankWordTokenizer()
detokenizer = TreebankWordDetokenizer()

text = 'besok saya bayar'
tokens = tokenizer.tokenize(text)
print(tokens)
detoken = detokenizer.tokenize(tokens)
print(detoken)