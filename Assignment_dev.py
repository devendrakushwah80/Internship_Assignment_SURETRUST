# 1. Custom Word Tokenizer (Medium)
# Create a tokenizer that:
# • Splits words based on whitespace.
# • Removes punctuation marks: . , ! ? : ;
# • Converts all tokens to lowercase.
# Input: "Hello, World! NLP is amazing."
# Expected Output:
# ['hello', 'world', 'nlp', 'is', 'amazing']

def custom_tokenizer(text):

    punct = ".,!?:;"
    
    # create translation table
    table = str.maketrans('', '', punct)
    
    text = text.translate(table)
    
    text = text.lower()
    
    tokens = text.split()
    
    return tokens


text = "Hello, World! NLP is amazing."
print(custom_tokenizer(text))



# 2. Subword Tokenization – WordPiece (Medium–Hard)
# Vocabulary:
# ["play", "##ing", "##er", "work", "##er", "##ing"]
# Tokenize the word:
# playing
# Expected Output:
# ["play", "##ing"]
# Rules:
# • Always choose the longest prefix in the vocabulary.
# • Remaining characters must start with ##.
# • If no match exists return [UNK].


def wordpiece_tokenize(word, vocab):

    tokens = []
    start = 0

    while start < len(word):

        end = len(word)
        cur_substr = None

        while start < end:
            
            substr = word[start:end]

            if start > 0:
                substr = "##" + substr

            if substr in vocab:
                cur_substr = substr
                break

            end -= 1

        if cur_substr is None:
            return ["[UNK]"]

        tokens.append(cur_substr)

        start = end

    return tokens


vocab = ["play", "##ing", "##er", "work", "##er", "##ing"]

print(wordpiece_tokenize("playing", vocab))

# 3. Byte Pair Encoding Merge Step (Hard)
# Corpus:
# low
# lower
# newest
# widest
# Step 1: Split into characters:
# l o w
# l o w e r
# n e w e s t
# w i d e s t
# Task:
# • Count pair frequencies
# • Merge the most frequent pair
# • Return updated corpus

from collections import defaultdict

corpus = ["low", "lower", "newest", "widest"]
corpus = [list(word) for word in corpus]


def count_pairs(corpus):

    pair_freq = defaultdict(int)

    for word in corpus:
        for i in range(len(word)-1):
            pair = (word[i], word[i+1])
            pair_freq[pair] += 1

    return pair_freq


def merge_pair(pair, corpus):

    merged = []

    for word in corpus:
        new_word = []
        i = 0

        while i < len(word):

            if i < len(word)-1 and (word[i], word[i+1]) == pair:
                new_word.append(word[i] + word[i+1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1

        merged.append(new_word)

    return merged


pair_freq = count_pairs(corpus)

print("Pair Frequencies:", dict(pair_freq))

best_pair = max(pair_freq, key=pair_freq.get)

print("Most Frequent Pair:", best_pair)

corpus = merge_pair(best_pair, corpus)

print("Updated Corpus:", corpus)

# 4. SentencePiece Style Tokenization (Hard)
# Vocabulary:
# [" deep", " learning", " is", "fun"]
# Tokenize:
# "deep learning is fun"
# Expected Output:
# [" deep"," learning"," is","fun"]
# Rules:
# • Add 
#  before each word.
# • Use longest vocabulary match.
# • Handle unknown words using [UNK]


def sentencepiece_tokenizer(text, vocab):
    tokens = []
    text = " " + text

    i = 0
    n = len(text)

    while i < n:

        match = None
        for j in range(n, i, -1):

            sub = text[i:j]

            if sub in vocab:
                match = sub
                break

        if match:
            tokens.append(match)
            i += len(match)

        else:
            
            if text[i] == " ":
                i += 1
            else:
                tokens.append("[UNK]")
                break

    return tokens


# vocabulary
vocab = [" deep", " learning", " is", "fun"]

# input
text = "deep learning is fun"

tokens = sentencepiece_tokenizer(text, vocab)

print(tokens)

# 5. Tokenizer With Position Tracking (Hard)
# Create a tokenizer that returns tokens with start and end indices.
# Input:
# "I love NLP"
# Expected Output:
# ("I",0,1)
# ("love",2,6)
# ("NLP",7,10)
# Constraints:
# • Ignore punctuation
# • Track exact character positions

import string

def tokenizer_with_positions(text):

    tokens = []
    i = 0
    n = len(text)

    while i < n:
        if text[i] in string.whitespace or text[i] in string.punctuation:
            i += 1
            continue
        start = i
        while i < n and text[i] not in string.whitespace and text[i] not in string.punctuation:
            i += 1
        end = i
        token = text[start:end]
        tokens.append((token, start, end))

    return tokens


text = "I love NLP"
result = tokenizer_with_positions(text)
for t in result:
    print(t)


# 6. Mini BERT Tokenizer (Hard)
# Build a tokenizer that performs:
# • Lowercasing
# • Word tokenization
# • WordPiece tokenization
# • Adds special tokens
# Input:
# "Playing football"
# Vocabulary:
# ["play","##ing","football"]
# Expected Output:
# ["[CLS]","play","##ing","football","[SEP]"]

def wordpiece_tokenize(word, vocab):
    
    tokens = []
    start = 0
    
    while start < len(word):
        
        end = len(word)
        match = None
        
        while start < end:
            
            substr = word[start:end]
            
            if start > 0:
                substr = "##" + substr
                
            if substr in vocab:
                match = substr
                break
                
            end -= 1
        
        if match is None:
            return ["[UNK]"]
        
        tokens.append(match)
        start = end
    
    return tokens


def mini_bert_tokenizer(text, vocab):
    text = text.lower()
    words = text.split()
    
    tokens = []
    for word in words:
        tokens.extend(wordpiece_tokenize(word, vocab))
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    return tokens


vocab = ["play","##ing","football"]
# Input
text = "Playing football"
result = mini_bert_tokenizer(text, vocab)
print(result)