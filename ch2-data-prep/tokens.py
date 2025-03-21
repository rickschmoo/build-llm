import re

from importlib.metadata import version
import tiktoken
print('tiktoken version: ', version('tiktoken'))

from tokenizer import SimpleTokenizerV1, SimpleTokenizerV2

# read in a file called "the-verdict.txt", report number of words it contains, and print the first 99 words
with open('the-verdict.txt', 'r', encoding="utf-8") as f:
    raw_text = f.read()
print('The total number of characters in the file is:', len(raw_text))
print('The first 99 characters in the file are:', raw_text[:99])


# Split text into words
# words = raw_text.split()
# print('The number of words in the file is:', len(words))
# print('The first 99 words in the file are:', words[:99])

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
print('The number of tokens in the file is:', len(preprocessed))
print('The first 99 tokens in the file are:', preprocessed[:99])

# Remove white spaces from tokens
preprocessed = [token for token in preprocessed if token.strip()]

print('The number of tokens after removing white spaces is:', len(preprocessed))
print('The first 99 tokens after removing white spaces are:', preprocessed[:99])

# create a list of unique tokens in alphabetical order and print count
all_tokens = sorted(set(preprocessed))
print('The number of unique tokens in the file is:', len(all_tokens))
print('The first 99 unique tokens in the file are:', all_tokens[:99])

# add system tokens
all_tokens.extend(["<|endoftext|>", "<|unk|>"])

# create a vocabulary
vocab = {token:integer for integer, token in enumerate(all_tokens)}
# print('The vocabulary is:')
# for token, integer in vocab.items():
#     print(integer, token)
print('The number of tokens in the vocabulary is:', len(vocab))

for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

# try out SimpleTokenizerV2
tokenizer = SimpleTokenizerV2(vocab)
text = """"It's the last he painted, you know," 
       Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)

text_decoded = tokenizer.decode(ids)
print(text_decoded)


text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace or in someunknownPlace."
text = " <|endoftext|> ".join((text1, text2))
test_ids = (tokenizer.encode(text))
print(test_ids)
test_decoded = tokenizer.decode(test_ids)
print(test_decoded)

# try out tiktoken
tt_tokenizer = tiktoken.get_encoding("gpt2")

# tt_integers = tt_tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# print(tt_integers)

# tt_strings = tt_tokenizer.decode(tt_integers)
# print(tt_strings)

# Exercise 1
# text_ex2 = "Akwirw ier"
# tt_integers_ex2 = tt_tokenizer.encode(text_ex2, allowed_special={"<|endoftext|>"})
# print(tt_integers_ex2)
# tt_strings_ex2 = tt_tokenizer.decode(tt_integers_ex2)
# print(tt_strings_ex2)

# IMPLEMENT USING BPE tokenizer
enc_text = tt_tokenizer.encode(raw_text)
print('The number of tokens in the file is:', len(enc_text))

#select interesting portion of text
enc_sample = enc_text[50:]

context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y:      {y}")

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tt_tokenizer.decode(context), "---->", tt_tokenizer.decode([desired]))


