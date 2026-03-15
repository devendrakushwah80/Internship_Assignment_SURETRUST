# NLP Tokenization Algorithms (SureTrust AIML Internship)

This repository contains implementations of important **Natural Language Processing (NLP) tokenization techniques** built from scratch using Python as part of the **SureTrust AIML Internship Program**.

Tokenization is a fundamental preprocessing step in NLP that converts raw text into tokens which machine learning and deep learning models can process.

---

## Implemented Algorithms

### 1. Custom Word Tokenizer

A simple tokenizer that:

* Splits text based on whitespace
* Removes punctuation (`. , ! ? : ;`)
* Converts text to lowercase

Example

Input:

```
Hello, World! NLP is amazing.
```

Output:

```
['hello', 'world', 'nlp', 'is', 'amazing']
```

---

### 2. WordPiece Tokenization

Implements the **WordPiece algorithm** used in transformer models like **BERT**.

Features:

* Longest prefix matching
* Subword tokenization
* Handles rare words using `[UNK]`

Example:

```
playing → ["play","##ing"]
```

---

### 3. Byte Pair Encoding (BPE)

Implements a simplified version of the **BPE algorithm** used in models like **GPT and RoBERTa**.

Steps:

* Split words into characters
* Count pair frequencies
* Merge the most frequent pair
* Update the corpus

---

### 4. SentencePiece Style Tokenization

Implements tokenization similar to **SentencePiece** used in **T5 and ALBERT**.

Features:

* Space-aware tokens
* Longest vocabulary matching
* Unknown token handling

Example:

```
deep learning is fun
→ [" deep"," learning"," is","fun"]
```

---

### 5. Tokenizer with Position Tracking

Returns tokens along with their **start and end positions** in the original text.

Example:

```
Input:
"I love NLP"

Output:
("I",0,1)
("love",2,6)
("NLP",7,10)
```

---

### 6. Mini BERT Tokenizer

A simplified version of the **BERT tokenization pipeline**.

Steps:

* Lowercase conversion
* Word tokenization
* WordPiece tokenization
* Add special tokens `[CLS]` and `[SEP]`

Example:

```
Input:
Playing football

Output:
["[CLS]","play","##ing","football","[SEP]"]
```
---

## Tech Stack

* Python
* NLP Preprocessing
* Tokenization Algorithms

---

## Learning Outcomes

* Understanding tokenization techniques used in modern NLP systems
* Implementing subword tokenization algorithms
* Building NLP preprocessing pipelines from scratch

---

## Internship

This project was completed as part of the **SureTrust AIML Internship Program**.

---

## Author

Devendra Kushwah
AI/ML Enthusiast
