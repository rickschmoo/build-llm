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