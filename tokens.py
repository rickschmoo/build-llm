import re

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
all_words = sorted(set(preprocessed))
print('The number of unique tokens in the file is:', len(all_words))
print('The first 99 unique tokens in the file are:', all_words[:99])

# create a vocabulary
vocab = {token:integer for integer, token in enumerate(all_words)}
print('The vocabulary is:')
for token, integer in vocab.items():
    print(integer, token)
print('The number of tokens in the vocabulary is:', len(vocab))

# create an inverse version of the vocabulary that maps token IDs back to the corresponding text tokens
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {integer: token for token, integer in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
    
tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know," 
       Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)

text_decoded = tokenizer.decode(ids)
print(text_decoded)

# random test
test = "I wonder, do you like tea?"
test_ids = (tokenizer.encode(test))
print(test_ids)
test_decoded = tokenizer.decode(test_ids)
print(test_decoded)
